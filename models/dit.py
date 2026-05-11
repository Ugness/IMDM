import os
import math
import typing

import einops
import flash_attn
import flash_attn.layers.rotary
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

env_val = os.getenv('DIT_USE_COMPILE', '0').lower()
USE_COMPILE = env_val in ['1', 'true', 'yes', 'on']
print(f"DIT: USE_COMPILE={USE_COMPILE}")

if USE_COMPILE:
    torch_compile_deco = torch.compile(model=None, mode=None, dynamic=False, options={"max_autotune": True, "triton.cudagraphs": False})
    jit_deco = lambda x: x
else:
    torch_compile_deco = lambda x: x
    jit_deco = torch.jit.script


# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

def bias_dropout_add_scale(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float,
        training: bool) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)

    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(
            x, bias, scale, residual, prob, training)

    return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
                         shift: torch.Tensor,
                         scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


@jit_deco
def bias_dropout_add_scale_fused_train(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(
        x, bias, scale, residual, prob, True)


@jit_deco
def bias_dropout_add_scale_fused_inference(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(
        x, bias, scale, residual, prob, False)


@jit_deco
def modulate_fused(x: torch.Tensor,
                                     shift: torch.Tensor,
                                     scale: torch.Tensor) -> torch.Tensor:
    return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
            # This makes the transformation on v an identity.
            self.cos_cached[:,:,2,:,:].fill_(1.)
            self.sin_cached[:,:,2,:,:].fill_(0.)

        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin,):
    with torch.amp.autocast('cuda', enabled=False):
        cos, sin = rotary_cos_sin
        cos = cos.to(qkv.dtype)
        sin = sin.to(qkv.dtype)
        cos = cos[0,:,0,0,:cos.shape[-1]//2]
        sin = sin[0,:,0,0,:sin.shape[-1]//2]
        q, k, v = qkv.chunk(3, dim=2)
        q = flash_attn.layers.rotary.apply_rotary_emb_torch(
            q.squeeze(dim=2), cos, sin)
        k = flash_attn.layers.rotary.apply_rotary_emb_torch(
            k.squeeze(dim=2), cos, sin)
        v = v.squeeze(dim=2)
    return q, k, v


def apply_rotary_pos_emb(qkv, cos, sin):
    cos = cos[0,:,0,0,:cos.shape[-1]//2]
    sin = sin[0,:,0,0,:sin.shape[-1]//2]
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


def regular_attention_multi_headed(q, k, v):
    # Assuming qkv is a tensor with shape [batch, seq_len, 3, num_heads, head_dim]
    # where the 3 represents Q, K, V packed in that order
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        attention_output = F.scaled_dot_product_attention(
            query=q.transpose(1, 2),
            key=k.transpose(1, 2),
            value=v.transpose(1, 2),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False)
    # [batch_size, seq_len, num_heads, head_dim]
    attention_output = attention_output.transpose(1, 2)
    return einops.rearrange(attention_output, 'b s h d -> b s (h d)')


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size
        
        # init_params
        torch.nn.init.normal_(self.mlp[0].weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.mlp[2].weight, mean=0.0, std=0.02)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                                            These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            - math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.bfloat16, device=t.device)
            / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding,
                 torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.
    
    Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes


    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings
        

#################################################################################
#                                 Core Model                                    #
#################################################################################

class DDiTBlockCausal(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True))
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, **kwargs):
        del kwargs
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        # attention operation
        x_skip = x
        x = self.norm1(x)

        qkv = self.attn_qkv(x)
        qkv = einops.rearrange(
            qkv,
            'b s (three h d) -> b s three h d',
            three=3,
            h=self.n_heads)
        with torch.amp.autocast('cuda', enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )
        qkv = einops.rearrange(qkv, 'b s ... -> (b s) ...')
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * seq_len,
            step=seq_len, dtype=torch.int32, device=qkv.device)
        x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0.0, causal=True)

        x = einops.rearrange(x, '(b s) h d -> b s (h d)',
                                                 b=batch_size)

        scale = torch.ones(1, device=x.device, dtype=x.dtype)
        x = bias_dropout_scale_fn(
            self.attn_out(x), None, scale, x_skip, self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(
            self.mlp(self.norm2(x)), None, scale, x, self.dropout)
        return x



class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, adaLN,
                             cond_dim=None, mlp_ratio=4,
                             dropout=0.1, use_qk_norm=False, softcap=None):
        super().__init__()
        self.n_heads = n_heads
        self.softcap = softcap
        self.adaLN = adaLN
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(dim//n_heads)
            self.k_norm = nn.LayerNorm(dim//n_heads)
    
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True))
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        if self.adaLN:
            self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()
        
        # init_params
        torch.nn.init.normal_(self.attn_qkv.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.attn_out.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.mlp[0].weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.mlp[2].weight, mean=0.0, std=0.02)


    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference


    def forward(self, x, rotary_cos_sin, c=None, seqlens=None, exclude_last_token=False):
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        x_skip = x
        x = self.norm1(x)

        if self.adaLN:
            # self.adaLN_modulation(c): (128, 1536)
            # self.adaLN_modulation(c)[:, None]: (128, 1, 1536)
            # "" .chunk(6, dim=2) returns 6 tuples of shapes (128, 1, 256)
            (shift_msa, scale_msa, gate_msa, shift_mlp,
             scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
            x = modulate_fused(x, shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = einops.rearrange(
            qkv,
            'b s (three h d) -> b s three h d',
            three=3,
            h=self.n_heads)
        orig_dtype = qkv.dtype
        if self.use_qk_norm:
            q,k,v = qkv.chunk(3, dim=2)
            q = self.q_norm(q)
            k = self.k_norm(k)
            qkv = torch.cat([q,k,v], dim=2).to(orig_dtype)
        with torch.amp.autocast('cuda', enabled=False):
            cos, sin = rotary_cos_sin
            if exclude_last_token:
                last_token = qkv[:, -1:]
                qkv = apply_rotary_pos_emb(
                    qkv[:, :-1], cos.to(qkv.dtype), sin.to(qkv.dtype)
                )
                qkv = torch.cat([qkv, last_token], dim=1)
            else:
                qkv = apply_rotary_pos_emb(
                    qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
                )
        x = flash_attn.flash_attn_qkvpacked_func(
            qkv, 0.0, causal=False,
            softcap=self.softcap,
            )

        x = einops.rearrange(x, 'b s h d -> b s (h d)',)

        if self.adaLN:
            x = bias_dropout_scale_fn(self.attn_out(x),
                                                                None,
                                                                gate_msa,
                                                                x_skip,
                                                                self.dropout)
            x = bias_dropout_scale_fn(
                self.mlp(modulate_fused(
                    self.norm2(x), shift_mlp, scale_mlp)),
                None, gate_mlp, x, self.dropout)
        else:
            scale = torch.ones(1, device=x.device, dtype=x.dtype)
            x = bias_dropout_scale_fn(
                self.attn_out(x), None, scale, x_skip, self.dropout)
            x = bias_dropout_scale_fn(
                self.mlp(self.norm2(x)), None, scale, x, self.dropout)
        return x


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.
    From "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains" and SDE paper.
    """
    def __init__(self, dim, scale=1.0, v=False):
        super().__init__()
        W = torch.randn(dim // 2) * scale
        self.v = v
        self.register_buffer('random_projection', W)

    def forward(self, x):
        # x: (B, N, D) will use the first dimension
        if self.v:
            D = x.shape[-1]
            x = x[..., :D//2]
            x_proj = x * self.random_projection.reshape(1, 1, -1) * 2 * math.pi
        else:
            x = x[..., :1]
            x_proj = x * self.random_projection.reshape(1, 1, -1) * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)    

class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim, mode='None', clip=False, is_imdm=False,
                 noise_type='randn', as_mdlm=False, num_mask=-1,
                 noise_dim=None, noise_scale=1.0):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))
        self.mode = mode
        self.clip = clip
        self.is_imdm = is_imdm
        self.noise_dim = noise_dim if noise_dim is not None else dim
        self.noise_type = noise_type
        self.vocab_dim = vocab_dim
        self.num_mask = num_mask
        if self.num_mask > 0:
            print('Using finite mask for IMDM')
            self.rand_fn = lambda x, *args, **kwargs: torch.randint(0, self.num_mask, size=x.shape[:2], device=x.device)
        elif self.noise_type == 'randn':
            print('Using randn noise for IMDM')
            self.rand_fn = lambda *args, **kwargs: torch.randn_like(*args, **kwargs) * noise_scale
        elif self.noise_type == 'learnable':
            print('Using learnable noise for IMDM')
            assert noise_scale == 1.0, "noise_scale must be 1.0 when using learnable noise"
            self.learnable_noise_mu = nn.Parameter(torch.zeros((1, 1, self.noise_dim)))
            self.learnable_noise_sigma = nn.Parameter(torch.ones((1, 1, self.noise_dim)))
            self.rand_fn = lambda *args, **kwargs: torch.randn_like(*args, **kwargs) * self.learnable_noise_sigma + self.learnable_noise_mu
        else:
            print('Using rand uniform noise for IMDM')
            self.rand_fn = lambda *args, **kwargs: (torch.rand_like(*args, **kwargs) * 2 - 1.) * noise_scale
            
        if self.is_imdm:
            # MDLM mode
            if as_mdlm:
                self.imdm_mlp = lambda x: torch.zeros_like(x)
            # Finite mask embedding mode
            elif num_mask > 0:
                self.imdm_mlp = nn.Embedding(num_mask, dim)
                torch.nn.init.zeros_(self.imdm_mlp.weight)
                if num_mask == 1:
                    self.imdm_mlp.weight.requires_grad = False # always zero
            # Infinite mask embedding mode
            else:
                self.imdm_mlp = nn.Sequential(
                        nn.Linear(in_features=noise_dim, out_features=dim*4),
                        nn.GELU(),
                        nn.Linear(dim*4, dim, bias=False) # [MASK] embedding will act as bias
                    )
                # truncated normal init
                nn.init.trunc_normal_(self.imdm_mlp[0].weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
                nn.init.zeros_(self.imdm_mlp[0].bias)
                nn.init.zeros_(self.imdm_mlp[2].weight)

    def forward(self, x, prob=None, indices=None, epsilon=None, reset=None, mask=None):
        if self.is_imdm:
            assert mask is not None, "mask_index indicator required for IMDM"
            embedding = self.embedding[x]
            if epsilon is None:
                # epsilon = torch.rand_like(embedding) - 1.
                # embedding: (B, L, dim)
                epsilon = self.rand_fn(torch.empty((*embedding.shape[:-1], self.noise_dim), dtype=embedding.dtype, device=embedding.device)) if self.noise_dim != embedding.shape[-1] else self.rand_fn(embedding)
                # epsilon: (B, L, dim) or (B, L) if num_mask > 0
            # elif self.noise_type == 'learnable':
            #     epsilon = self.learnable_noise_mu + self.learnable_noise_sigma * epsilon
            if reset is None:
                reset = torch.ones_like(x, dtype=torch.bool)
            if self.num_mask > 0:
                assert epsilon.dtype == torch.int64
                assert torch.all((epsilon >= 0) & (epsilon < self.num_mask)), f"epsilon values should be in [0, {self.num_mask-1}] for num_mask={self.num_mask}"
                epsilon = torch.where(reset.bool(), self.rand_fn(epsilon), epsilon)
                epsilon = epsilon.long()
                mask_embedding = self.imdm_mlp(epsilon) # (B, L, dim)
                mask = mask[:,:,None].to(embedding.dtype)
                embedding = embedding + mask_embedding * mask
            else:
                reset = reset.to(epsilon.dtype)[:,:,None]            
                epsilon = epsilon * (1 - reset) + self.rand_fn(epsilon) * reset
                mask_embedding = self.imdm_mlp(epsilon) # (B, L, dim)
                mask = mask[:,:,None].to(embedding.dtype)
                embedding = embedding + mask_embedding * mask
            return embedding, epsilon

        assert not self.is_imdm, "IMDM should not reach here"
        assert mask is None, "mask_index indicator only for IMDM"

        if x.ndim == 2 and prob is None and indices is None:
            return self.embedding[x]
        if prob is not None and indices is not None:
            dtype = self.embedding.dtype
            prob = prob.to(dtype)
            # prob: (B, L, k)
            # indices: (B, L, k)
            # get embedding for each index. build a (B, L, k, dim) tensor
            emb = self.embedding[indices]  # (B, L, k, dim)
            # weight each embedding by prob
            emb = emb * prob[..., None]  # (B, L, k, dim)
            # sum over k
            embedding = emb.sum(dim=2)  # (B, L, dim)
            return embedding
        if x.ndim == 3:
            prob = F.softmax(x, dim=-1)  # (B, L, V)
            dtype = self.embedding.dtype
            prob = prob.to(dtype)
            embedding = torch.einsum(
                'blv,vd->bld', prob, self.embedding)  # (B, L, dim)
            return embedding


class DDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim,
                             adaLN):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        
        
        self.adaLN = adaLN
        if self.adaLN:
            self.adaLN_modulation = nn.Linear(cond_dim,
                            2 * hidden_size,
                            bias=True)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()


    def forward(self, x, c):
        x = self.norm_final(x)
        if self.adaLN:
            shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
            x = modulate_fused(x, shift, scale)
        x = self.linear(x)
        return x


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)
        self.causal = config.algo.causal_attention
        self.adaLN = not self.causal
        self.config = config
        self.vocab_size = vocab_size
        self.mode = config.model.embedding_mode
        self.embedding_clip = getattr(config.model, "embedding_clip", False)
        dim = config.model.hidden_size
        cond_dim = config.model.cond_dim
        self.is_imdm = config.model.get("is_imdm", False)
        self.vocab_embed = EmbeddingLayer(dim, vocab_size, mode=self.mode, clip=self.embedding_clip,
                                          is_imdm=self.is_imdm, noise_type=config.algo.get("noise_type", "None"),
                                          as_mdlm=config.algo.get("as_mdlm", False),
                                          num_mask=config.algo.get("num_mask", -1),
                                          noise_dim=config.model.get("noise_dim", dim),
                                          noise_scale=config.algo.get("noise_scale", 1.0)
                                          )
        self.qk_norm = getattr(config.model, "qk_norm", False)
        self.softcap = getattr(config.model, "softcap", -1.) # <0.0 means no softcap
        if not self.causal:
            self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(dim // config.model.n_heads)

        blocks = []
        for _ in range(config.model.n_blocks):
            if self.causal:
                block = DDiTBlockCausal(
                    dim=dim,
                    n_heads=config.model.n_heads,
                    dropout=config.model.dropout)
            else:
                block = DDiTBlock(
                    dim=dim,
                    n_heads=config.model.n_heads,
                    cond_dim=cond_dim,
                    adaLN=self.adaLN,
                    dropout=config.model.dropout,
                    use_qk_norm=self.qk_norm,
                    softcap=self.softcap)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDiTFinalLayer(
            hidden_size=dim,
            out_channels=vocab_size,
            cond_dim=cond_dim,
            adaLN=self.adaLN)
        self.scale_by_sigma = config.model.scale_by_sigma
        
        if "is_di4c" in config:
            self.is_di4c = config.is_di4c
        else:
            self.is_di4c = config.is_di4c = False
            
        if "is_di4c_deterministic" in config:
            self.is_di4c_deterministic = config.is_di4c_deterministic
        else:
            self.is_di4c_deterministic = config.is_di4c_deterministic = False
                
        if self.is_di4c:
            print("Using Di4C")
            # Added for Di4C:
            self.latent_feature_dim = 128
            self.latent_projection = nn.Sequential(
                nn.Linear(in_features=self.latent_feature_dim, out_features=self.latent_feature_dim*4),
                nn.GELU(),
                nn.Linear(self.latent_feature_dim*4, config.model.hidden_size)
            )

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    @torch_compile_deco
    def forward(self, x, sigma, prob=None, indices=None, epsilon=None, reset=None, mask=None):
        if self.is_imdm:
            x, epsilon = self.vocab_embed(x, prob=prob, indices=indices, epsilon=epsilon, reset=reset, mask=mask)
        else:
            x = self.vocab_embed(x, prob=prob, indices=indices)
        if self.causal:
            t_cond = None
        else:
            t_cond = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x)

        if self.is_di4c: # Di4C
            if self.is_di4c_deterministic:
                z = torch.ones(x.size(0)).to(x.device) * 0.5
            else:
                z = torch.rand(x.size(0)).to(x.device)
            z_emb = transformer_timestep_embedding(
                    z.view(-1) * 1000, self.latent_feature_dim
            )
            z_emb = self.latent_projection(z_emb)
            x = torch.cat([x, z_emb[:,None,:]], dim=1)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c=t_cond, seqlens=None, exclude_last_token=self.is_di4c)
            x = self.output_layer(x, c=t_cond)
        
        if self.is_di4c:
            # Di4C
            x = x[:, :-1, :]
        if self.is_imdm:
            return x, epsilon
        return x

# From https://github.com/yang-song/score_sde_pytorch/ which is from
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
def transformer_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
        assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
        half_dim = embedding_dim // 2
        # magic number 10000 is from transformers
        emb = math.log(max_positions) / (half_dim - 1)
        # emb = math.log(2.) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
        # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
                emb = F.pad(emb, (0, 1), mode='constant')
        assert emb.shape == (timesteps.shape[0], embedding_dim)
        return emb