import os
import collections
import copy
import pickle
from functools import partial
import math
import time

import fsspec
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import OmegaConf, open_dict

import trainer_base
import models
import utils

from utils import (alpha_to_gamma, gamma_to_alpha,
                   build_luts, build_luts_ext,
                   pull_lut, scalable_topk_approximation,
                   to_duo_gamma,
                   robust_seeded_randn)

class AR(trainer_base.TrainerBase):
    def __init__(self, config, tokenizer):
        vocab_size = tokenizer.vocab_size
        if (not hasattr(tokenizer, 'mask_token')
                or tokenizer.mask_token is None):
            self.mask_index = vocab_size
            vocab_size += 1
        else:
            self.mask_index = tokenizer.mask_token_id
        super().__init__(config, tokenizer,
                                         vocab_size=vocab_size)
        self.save_hyperparameters()
        self._validate_configuration()

    def _validate_configuration(self):
        super()._validate_configuration()
        assert not self.config.algo.time_conditioning
        assert self.config.prior.type == 'none'

    def _process_model_input(self, x0, valid_tokens):
        input_tokens = x0[:, :-1]
        output_tokens = x0[:, 1:]
        valid_tokens = valid_tokens[:, 1:]
        return input_tokens, output_tokens, valid_tokens

    def nll(self, input_tokens, output_tokens,
                    current_accumulation_step):
        del current_accumulation_step
        output = self.backbone(input_tokens, None)
        output[:, :, self.mask_index] = self.neg_infinity
        output = output.log_softmax(-1)
        return - output.gather(
            -1, output_tokens[:, :, None])[:, :, 0]

    def generate_samples(self, num_samples, **kwargs):
        # precompute token buffer
        num_pred_tokens = self.num_tokens - 1
        x = torch.zeros(
            (num_samples, num_pred_tokens + 1),
            dtype=torch.long,
            device=self.device)
        x[:, 0] = self.tokenizer.bos_token_id
        # precompute noise
        noise = (torch.distributions.Gumbel(0, 1)
                         .sample((num_samples, num_pred_tokens, self.vocab_size))
                         .to(self.device))
        if self.config.sampling.use_float64:
            noise = noise.to(torch.float64)
        for i in range(num_pred_tokens):
            output = self.backbone(x[:, :i + 1], None)
            output[:, :, self.mask_index] = self.neg_infinity
            output = output.log_softmax(-1)
            y = (output[:, -1, :] + noise[:, i, :]).argmax(-1)
            x[:, i + 1] = y
        return x

    def _process_sigma(self, sigma):
        del sigma
        return None


class MDLM(trainer_base.AbsorbingState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def _validate_configuration(self):
        # ancestral sampling isn't desirable because it's slow
        assert self.sampler == 'ancestral_cache'

    def _process_model_output(self, model_output, xt, sigma):
        del sigma

        # SUBS masking
        mask_idx_tensor = torch.tensor([self.mask_index], device=model_output.device)
        model_output = model_output.index_fill(dim=-1, index=mask_idx_tensor, value=self.neg_infinity)
        
        if xt.ndim == 3:
            xt = xt.argmax(-1)            
        
        model_output = model_output.log_softmax(dim=-1)
        
        # Carry Over
        unmasked_mask = (xt != self.mask_index).unsqueeze(-1)
        replacement = torch.full_like(model_output, self.neg_infinity)
        replacement = torch.scatter(replacement, -1, xt.unsqueeze(-1), torch.zeros_like(replacement[..., :1]))
        model_output = torch.where(unmasked_mask, replacement, model_output)
        return model_output

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                                        dalpha_t, low_var=False, **kwargs):
        del xt
        log_p_theta = torch.gather(
            input=log_x_theta,
            dim=-1,
            index=x0[:, :, None]).squeeze(-1)
        return log_p_theta * dalpha_t / (1 - alpha_t)

    def _get_score(self, x, sigma):
        model_output = self.forward(x, sigma)
        # score(x, t) = p_t(y) / p_t(x)
        # => log score(x, t) = log p_t(y) - log p_t(x)
        
        # case 1: x = masked
        #   (i) y = unmasked
        #     log score(x, t) = log p_\theta(x)|_y + log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))
        #   (ii) y = masked
        #     log score(x, t) = 0

        # case 2: x = unmasked
        #   (i) y != masked, y != x
        #     log score(x_i, t) = - inf
        #   (ii) y = x 
        #     log score(x_i, t) = 0
        #   (iii) y = masked token
        #     log score(x_i, t) = - log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))
        
        log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
        assert log_k.ndim == 1
        
        masked_score = model_output + log_k[:, None, None]
        masked_score[:, :, self.mask_index] = 0

        unmasked_score = self.neg_infinity * torch.ones_like(
            model_output)
        unmasked_score = torch.scatter(
            unmasked_score,
            -1,
            x[..., None],
            torch.zeros_like(unmasked_score[..., :1]))
        unmasked_score[:, :, self.mask_index] = - (
            log_k[:, None] * torch.ones_like(x))
        
        masked_indices = (x == self.mask_index).to(
            model_output.dtype)[:, :, None]
        model_output = (
            masked_score * masked_indices
            + unmasked_score * (1 - masked_indices))
        return model_output.exp()


class D3PMAbsorb(trainer_base.AbsorbingState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def _validate_configuration(self):
        super()._validate_configuration()
        assert self.noise.type == 'log-linear'
        assert self.parameterization == 'mean'

    def _process_model_output(self, model_output, xt, sigma):
        del xt 
        del sigma
        if self.subs_masking:
            model_output[:, :, self.mask_index] += self.neg_infinity
        return model_output.log_softmax(dim=-1)

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                                        dalpha_t, low_var=False):
        del dalpha_t
        assert not low_var
        dt = 1 / self.T
        t = 1 - alpha_t  # Only valid for log-linear schedule.
        t = t.clamp(0., 1.0 - 1e-4)
        alpha_t = alpha_t + torch.zeros_like(xt)
        alpha_s = t - dt + torch.zeros_like(xt)
        assert alpha_s.shape == xt.shape
        assert alpha_t.shape == xt.shape
        log_x_theta_at_x0 = torch.gather(
            log_x_theta, -1, x0[:, :, None]).squeeze(-1)
        log_x_theta_at_m = log_x_theta[:, :, self.mask_index]
        x_theta_at_m = log_x_theta_at_m.exp()
        
        term_1_coef = dt / t
        term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
        term_1_log_dr = log_x_theta_at_x0
        
        term_2_coef = 1 - dt / t
        term_2_log_nr = term_1_log_nr
        term_2_log_dr = torch.log(
            alpha_s * x_theta_at_m / (t - dt) + 1)
        L_vb_masked = (
            term_1_coef * (term_1_log_nr - term_1_log_dr)
            + term_2_coef * (term_2_log_nr - term_2_log_dr))

        diffusion_loss = self.T * L_vb_masked * (xt == self.mask_index)
        return self._reconstruction_loss(x0) + diffusion_loss


class SEDDAbsorb(trainer_base.AbsorbingState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def _validate_configuration(self):
        super()._validate_configuration()
        assert self.config.sampling.predictor == 'analytic'

    def _get_score(self, x, sigma):
        return self.forward(x, sigma).exp()

    def _process_model_output(self, model_output, xt, sigma):
        esigm1_log = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            sigma.exp() - 1).log().to(model_output.dtype)
        # logits shape
        # (batch_size, context_length, vocab_size)
        model_output = (model_output
                                        - esigm1_log[:, None, None]
                                        - np.log(model_output.shape[-1] - 1))
        # The below scatter operation sets the log score
        # for the input word to 0.
        model_output = torch.scatter(
            model_output, -1, xt[..., None],
            torch.zeros_like(model_output[..., :1]))
        return model_output

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                                        dalpha_t, low_var=False):
        """Computes the SEDD loss for the Absorbing State Diffusion.

        Args:
            log_x_theta: float torch.Tensor with shape (batch_size,
                    context_length, vocab_size),
                    log score, output of the denoising network.
            xt: int torch.Tensor with shape (batch_size,
                    context_length), input.
            x0: int torch.Tensor with shape (batch_size,
                    context_length), input.
            alpha_t: float torch.Tensor with shape (batch_size, 1),
                    signal level.
            alpha_t: float torch.Tensor with shape (batch_size, 1),
                    signal level.
            dalpha_t: float or float torch.Tensor with shape (batch_size, 1),
                    time derivative of signal level.
            low_var: bool, low variance loss during training.
        
        Returns:
            loss with shape (batch_size, context_length).
        """
        assert not low_var
        masked_indices = xt == self.mask_index
        sigma = self._sigma_from_alphat(alpha_t)
        dsigma = - dalpha_t / alpha_t

        expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
        q_ratio = 1 / expsig_minus_1[masked_indices]

        words_that_were_masked = x0[masked_indices]

        neg_term = q_ratio * torch.gather(
            log_x_theta[masked_indices],
            -1,
            words_that_were_masked[..., None]).squeeze(-1)
        score = log_x_theta[masked_indices].exp()
        if self.mask_index == self.vocab_size - 1:
            pos_term = score[:, :-1].sum(dim=-1)
        else:
            pos_term = score[:, : self.mask_index].sum(
                dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
        const = q_ratio * (q_ratio.log() - 1)

        entropy = torch.zeros(* xt.shape, device=xt.device)
        entropy[masked_indices] += pos_term - neg_term + const
        return dsigma * entropy


class DUO_BASE(trainer_base.UniformState):
    def __init__(self, config, tokenizer, **kwargs):
        super().__init__(config, tokenizer, **kwargs)
        self._validate_configuration()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = collections.OrderedDict(
            (k, v) for k, v in checkpoint['state_dict'].items()
            if not k.startswith('teacher'))
        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = collections.OrderedDict(
            (k, v) for k, v in checkpoint['state_dict'].items()
            if not k.startswith('teacher'))
        super().on_load_checkpoint(checkpoint)

    def _process_model_output(self, model_output, xt, sigma):
        del xt, sigma
        return model_output.log_softmax(dim=-1)

    def _compute_posterior(self, x, xt, alpha_s, alpha_t):
        """Computes the posterior / approximate posterior.

        Args:
            x: Either clean input `x0` (one-hot),
                or model's predicted `x_theta` of shape (B, L, V).
            xt: The noisy latent (as indices) of shape (B, L).
            alpha_s: Noise level at s of shape (B, [L | 1], 1).
            alpha_t: Noise level at t of shape (B, [L | 1], 1).

        Returns:
            Posterior / approximate posterior of shape (B, L, V).
        """
        if self.config.sampling.use_float64:
            x = x.to(torch.float64)
        if alpha_s.ndim == 2:
            alpha_s = alpha_s.unsqueeze(-1)
        if alpha_t.ndim == 2:
            alpha_t = alpha_t.unsqueeze(-1)
        alpha_ts = alpha_t / alpha_s
        d_alpha = alpha_s - alpha_t
        xt_one_hot = F.one_hot(xt, self.vocab_size).to(
            self.dtype).to(self.device)
        return (
            (alpha_t * self.vocab_size * x * xt_one_hot + (
                alpha_ts - alpha_t) * xt_one_hot + d_alpha * x + (
                    1 - alpha_ts) * (1 - alpha_s) / self.vocab_size) / (
                        alpha_t * self.vocab_size * torch.gather(
                            x, -1, xt[..., None]) + (1 - alpha_t)))

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                                        dalpha_t, low_var=False): # Computes Eq 5.
        assert alpha_t.ndim == 2
        assert x0.ndim == 2
        assert xt.ndim == 2
        if torch.is_tensor(dalpha_t) and dalpha_t.ndim == 1:
            dalpha_t = dalpha_t.unsqueeze(-1)
        assert not torch.is_tensor(dalpha_t) or dalpha_t.ndim == 2
        x_reconst = log_x_theta.exp()
        x_bar_theta = self.vocab_size * alpha_t[
                :, :, None] * x_reconst + 1 - alpha_t[:, :, None]
        coeff = dalpha_t / (self.vocab_size * alpha_t)
        x_eq_xt = (x0 == xt).float()
        x_neq_xt = 1 - x_eq_xt
        xbar_xt = (1 - alpha_t) + self.vocab_size * alpha_t * x_eq_xt
        xbar_theta_xt = torch.gather(
            x_bar_theta, -1, xt.unsqueeze(-1)).squeeze(-1)
        xbar_theta_x = torch.gather(
            x_bar_theta, -1, x0.unsqueeze(-1)).squeeze(-1)
        term1 = self.vocab_size * (1 / xbar_xt
                                                                - 1 / xbar_theta_xt) # Eq 5. term 1
        
        const = (1 - alpha_t) / (self.vocab_size * alpha_t
                                                         + 1 - alpha_t)
        term2_coefs = x_eq_xt * const + x_neq_xt
        term2_offset = ((self.vocab_size - 1) * const * x_eq_xt
                                        - (1 / const) * x_neq_xt) * const.log()
        term2_theta = - term2_coefs * (
            x_bar_theta.log().sum(-1)
            - self.vocab_size * xbar_theta_xt.log())
        term2_theta = (
            term2_theta
            - self.vocab_size * alpha_t / (1 - alpha_t) * (
                xbar_theta_x.log() - xbar_theta_xt.log()) * x_neq_xt)
        term2 = term2_theta + term2_offset
        diffusion_loss = coeff * (term1 - term2)
        assert diffusion_loss.ndim == 2
        return diffusion_loss

    def _ancestral_update(self, x, t, dt, p_x0=None,
                                     noise_removal_step=False):
        del p_x0
        _, alpha_t = self.noise(t)
        if noise_removal_step:
            alpha_s = torch.ones_like(alpha_t)
        else:
            _, alpha_s = self.noise(t - dt)
        sigma_t = self._sigma_from_alphat(alpha_t)
        assert alpha_t.ndim == 2
        
        q_xs = self._compute_posterior(
            x=self.forward(x, sigma_t).exp(),
            xt=x,
            alpha_s=alpha_s,
            alpha_t=alpha_t)
        if self.p_nucleus < 1:
            q_xs = utils.top_k_top_p_filtering(
                q_xs.log(), top_p=self.p_nucleus)
        return None, trainer_base.sample_categorical(q_xs, self.config.sampling.temperature)


class Integral(torch.autograd.Function):
    """
    torch module calculating UDLM's p_t 
    """

    @staticmethod
    def forward(ctx, gamma_t, data):
        gamma_max = data['gamma_max']
        gamma_min = data['gamma_min']
        if (gamma_t.max() > gamma_max) or (
            gamma_t.min() < gamma_min):
            print('max:{} {}'.format(gamma_t.max(), gamma_max))
            print('min:{} {}'.format(gamma_t.min(), gamma_min))
            gamma_t = torch.clip(gamma_t, gamma_min, gamma_max)
        indices = torch.round(
            (data['num_points'] - 1) * (gamma_t - gamma_min) / (
                    gamma_max - gamma_min)).long()
        grad_pt = data['grad_pt']
        ctx.grad_pt = grad_pt[indices]
        
        pt = data['pt'][indices]
        assert pt.shape == gamma_t.shape
        return pt

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.grad_pt * grad_output, None


class DUO(DUO_BASE):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        with fsspec.open(self.config.algo.integral_cache_path,
                                         'rb') as f:
            self.integral_cache = pickle.load(f)
        self.integral_cache['pt'] = torch.from_numpy(
            self.integral_cache['pt'])
        self.integral_cache['grad_pt'] = torch.from_numpy(
            self.integral_cache['grad_pt'])
        self.gamma_min = self.config.algo.gamma_min
        self.gamma_max = self.config.algo.gamma_max
        self.gumbel_tau_log10_start = self.config.algo.gumbel_tau_log10_start
        self.gumbel_tau_log10_end = self.config.algo.gumbel_tau_log10_end
        self.curriculum_start = self.config.algo.curriculum_start
        self.curriculum_end = self.config.algo.curriculum_end
        self.loss_type = self.config.algo.loss_type
        self._validate_configuration()

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.integral_cache['pt'] = self.integral_cache[
            'pt'].to(*args, **kwargs)
        self.integral_cache['grad_pt'] = self.integral_cache[
            'grad_pt'].to(*args, **kwargs)
        return self

    def _compute_gumbel_tau_inverse(self):
        start = self.gumbel_tau_log10_start
        end = self.gumbel_tau_log10_end
        delta = end - start
        if self.global_step < self.curriculum_start:
            tau = start
        elif self.global_step < self.curriculum_end:
            frac = (self.global_step - self.curriculum_start) / (
                self.curriculum_end - self.curriculum_start)
            tau = start + frac * delta
        else:
            tau = -10
        return 10 ** (-tau)

    def training_step(self, batch, batch_idx):
        self.log(name='gumbel_tau_log10',
                         value=1 / self._compute_gumbel_tau_inverse(),
                         on_step=True,
                         on_epoch=False,
                         sync_dist=True)
        return super().training_step(batch, batch_idx)

    def _gamma_to_alphat(self, gamma_t): # eq 10.
        integral = Integral.apply(gamma_t, self.integral_cache)
        return (self.vocab_size * integral - 1) / (
            self.vocab_size - 1)

    def _prior_loss(self):
        alpha_1 = self._gamma_to_alphat(
            torch.tensor(self.gamma_max))
        loss = ((alpha_1 + (1 - alpha_1) / self.vocab_size) * torch.log(
            (self.vocab_size - 1) * alpha_1 + 1) + (
                1 - 1 / self.vocab_size) * (1 - alpha_1) * torch.log(1 - alpha_1))
        return loss.item()

    def _q_xt_gaussian(self, x, gamma_t):
        """Computes the noisy sample xt."""
        assert gamma_t.ndim == 1
        assert x.ndim == 3
        gamma_t = gamma_t.unsqueeze(-1).unsqueeze(-1)
        alpha_t = torch.sigmoid(-gamma_t).sqrt()
        sigma_t = torch.sigmoid(gamma_t).sqrt()
        epsilon = torch.randn(x.shape, dtype=torch.float32,
                                                    device=self.device)
        return alpha_t * x + sigma_t * epsilon

    def nll(self, x0, output_tokens,
                    current_accumulation_step=None, train_mode=False, xT=None, **kwargs):
        # TODO: use xT
        use_true_nll = (self.global_step > self.curriculum_end
                                        or not train_mode)
        if use_true_nll:
            return super().nll(x0, output_tokens,
                                                 current_accumulation_step)
        del output_tokens
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        gamma_t = self.gamma_min + t * (self.gamma_max
                                                                        - self.gamma_min)    
        gamma_t_prime = self.gamma_max - self.gamma_min
        alpha_t = self._gamma_to_alphat(gamma_t)
        T = 1000
        dalpha_t = gamma_t_prime * T * (
            self._gamma_to_alphat(gamma_t + 1 / T) - alpha_t)
        alpha_t = alpha_t.unsqueeze(-1)
        dalpha_t = dalpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(alpha_t)

        x0_one_hot = F.one_hot(x0, self.vocab_size)
        xt = self._q_xt_gaussian(x0_one_hot, gamma_t)
        xt = xt * self._compute_gumbel_tau_inverse()
        xt_usdm = xt.argmax(-1)
        log_x_theta = self.forward(xt, sigma=sigma)

        return self.nll_per_token(log_x_theta=log_x_theta,
                                                            xt=xt_usdm,
                                                            x0=x0,
                                                            alpha_t=alpha_t,
                                                            dalpha_t=dalpha_t,
                                                            low_var=False)


class Distillation(DUO):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.update_teacher_every = config.algo.update_teacher_every
        self.save_hyperparameters()
        self.teacher = None
        self.teacher_ema = config.algo.teacher_ema
        self.linear_growth_dt = config.algo.linear_growth_dt
        self.linear_growth_min = config.algo.linear_growth_min
        self.linear_growth_max = config.algo.linear_growth_max

    def _validate_configuration(self):
        assert os.path.exists(
            self.config.algo.integral_cache_path), (
                'The integral cache (Eq. 10 in the paper) for '
                f'the {self.config.data.tokenizer_name_or_path} '
                ' tokenizer doesnt exist at '
                f'{self.config.algo.integral_cache_path}. '
                'Please generate it by running the utils.py script, '
                'and ensure the correct path is specified using the '
                'algo.integral_cache_path flag.')
        assert self.loss_type in {
            'kl-fwd', 'kl-bwd', 'posterior', 'kl-posterior'}

    def _maybe_update_teacher_weights(self):
        if self.global_step % self.update_teacher_every != 0:
            return
        if self.teacher_ema:
            self.ema.copy_to(self.teacher.parameters())
        else:
            for better_param, current_param in zip(
                self.backbone.parameters(), self.teacher.parameters()):
                if current_param.requires_grad:
                    current_param.data.copy_(better_param.data)

    @torch.no_grad()
    def _teacher_logits(self, xt, sigma):
        if self.teacher is None:
            self.teacher = copy.deepcopy(self.backbone)
        self._maybe_update_teacher_weights()

        sigma = self._process_sigma(sigma)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            model_output = self.teacher(xt, sigma)
        logits = self._process_model_output(
            model_output=model_output, xt=xt, sigma=sigma)
        return logits.detach()

    def _sample_trajectory(self, x0, gamma_t, gamma_s):
        """Computes the noisy sample xt."""
        assert gamma_t.ndim == 1
        assert gamma_s.ndim == 1
        assert x0.ndim == 2
        x0 = F.one_hot(x0, self.vocab_size).to(
            self.dtype).to(self.device)
        gamma_t = gamma_t.unsqueeze(-1).unsqueeze(-1)
        alpha_t = torch.sigmoid(-gamma_t).sqrt()
        sigma_t = torch.sigmoid(gamma_t).sqrt()

        gamma_s = gamma_s.unsqueeze(-1).unsqueeze(-1)
        alpha_s = torch.sigmoid(-gamma_s).sqrt()
        sigma_s = torch.sigmoid(gamma_s).sqrt()
        
        epsilon = torch.randn(x0.shape, dtype=torch.float32,
                                                    device=self.device)
        xt = alpha_t * x0 + sigma_t * epsilon
        xs = alpha_s * x0 + sigma_s * epsilon
        return xt, xs

    def _compute_dt(self):
        if self.linear_growth_dt:
            scale = self.global_step / self.trainer.max_steps
            return self.linear_growth_min + scale * (
                self.linear_growth_max -  self.linear_growth_min)
        n = self.global_step // self.update_teacher_every
        return 2 ** n / self.T

    def nll(self, x0, output_tokens,
                    current_accumulation_step=None, train_mode=None, xT=None):
        # TODO: use xT
        del output_tokens, train_mode
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        dt = self._compute_dt()
        t = torch.clip(t + dt, 0, 1)

        gamma_t = self.gamma_min + t * (self.gamma_max - self.gamma_min)
        gamma_s = self.gamma_min + (
            t - dt) * (self.gamma_max - self.gamma_min)

        alpha_t = self._gamma_to_alphat(gamma_t)
        alpha_t = alpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        usdm_alpha_s = self._gamma_to_alphat(gamma_s)
        usdm_alpha_s = usdm_alpha_s.unsqueeze(-1)
        assert usdm_alpha_s.ndim == 2

        xt, xs = self._sample_trajectory(x0, gamma_t, gamma_s)
        xt_discrete = xt.argmax(-1)
        xs_discrete = xs.argmax(-1)
        log_x_theta_student = self.forward(
            xt_discrete, sigma=self._sigma_from_alphat(alpha_t))
        log_x_theta_teacher = self._teacher_logits(
            xs_discrete, sigma=self._sigma_from_alphat(usdm_alpha_s))
        if self.config.training.loss_precision == 'float64':
            log_x_theta_student = log_x_theta_student.to(torch.float64)
            log_x_theta_teacher = log_x_theta_teacher.to(torch.float64)
        if self.loss_type == 'kl-fwd':
            return (log_x_theta_teacher.exp() * (
                log_x_theta_teacher - log_x_theta_student)).sum(-1)
        elif self.loss_type == 'kl-bwd':
            return (log_x_theta_student.exp() * (
                log_x_theta_student - log_x_theta_teacher)).sum(-1)
        
    def training_step(self, batch, batch_idx):
        self.log(name='dt',
                         value=self._compute_dt(),
                         on_step=True,
                         on_epoch=False,
                         sync_dist=True)
        return super().training_step(batch, batch_idx)


class Rectification(DUO): # Training as duo, without curriculum
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.save_hyperparameters()
        self.use_linear_schedule = config.algo.use_linear_schedule
        self.use_simple_loss = config.algo.use_simple_loss
        self.onestep_mode = config.algo.onestep_mode
        self.debug = getattr(config.algo, 'debug', False)
    
    def _compute_gumbel_tau_inverse(self):
        return 1e-10

    def q_xt(self, x0, alpha_t):
        """Computes the noisy sample xt."""
        assert alpha_t.ndim == 2
        assert x0.ndim == 2
        alpha_t = alpha_t
        xT = self.prior_sample(x0.shape[0], x0.shape[1])
        xt = torch.where(
            torch.rand_like(x0, dtype=torch.float32) <= alpha_t,
            x0,
            xT)
        return xt

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                                        dalpha_t, low_var=False, simple_loss=False):
        if simple_loss:
            loss = F.cross_entropy(
                log_x_theta.view(-1, self.vocab_size),
                x0.view(-1),
                reduction='none')
            loss = loss.view(xt.shape)
            return loss
        else:
            return super().nll_per_token(
                            log_x_theta=log_x_theta,
                            xt=xt,
                            x0=x0,
                            alpha_t=alpha_t,
                            dalpha_t=dalpha_t,
                            low_var=low_var
                            )

    def nll(self, x0, output_tokens,
                    current_accumulation_step=None, train_mode=False, xT=None, given_t=None, not_sampling_t=False):
        del output_tokens
        if given_t is not None:
            t = given_t
        else:
            raise NotImplementedError
        assert t.shape[0] == x0.shape[0]
        if self.T > 0:
            assert 0
        
        dalpha_t, alpha_t = self.noise(t)        
        alpha_t = alpha_t.unsqueeze(-1)
        dalpha_t = dalpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(alpha_t)

        if given_t is not None and xT is not None:
            xt = xT
        else:
            raise NotImplementedError
        log_x_theta = self.forward(xt, sigma=sigma)
        return self.nll_per_token(log_x_theta=log_x_theta,
                                    xt=xt,
                                    x0=x0,
                                    alpha_t=alpha_t,
                                    dalpha_t=dalpha_t,
                                    low_var=train_mode and self.loss_type == 'low_var',
                                    simple_loss=self.use_simple_loss,
                                    )

def scale_logits(logits, cap_value=30.0):
    if cap_value <= 0:
        return logits
    return cap_value * torch.tanh(logits / cap_value)


class IMDM(MDLM):
    def __init__(self, config, tokenizer):
        self.as_mdlm = config.algo.get('as_mdlm', False)
        with open_dict(config):
            config.model.is_imdm = True
            config.model.as_mdlm = self.as_mdlm
            
        super().__init__(config, tokenizer)
        self.alpha_min = self.config.algo.alpha_min
        self.alpha_max = self.config.algo.alpha_max
        self.noise_type = config.algo.noise_type
        self.num_mask = config.algo.get('num_mask', -1)
        assert self.noise_type in ['rand', 'randn', 'learnable']
        self.noise_scale = config.algo.get('noise_scale', 1.0)
        
        if self.num_mask > 0:
            self.rand_fn = lambda *args, **kwargs: torch.randint(0, self.num_mask, args[:2], **kwargs)
            def robust_seeded_randint(*args, **kwargs):
                uniform_noise = robust_seeded_randn(*args, mode='rand', **kwargs)
                if uniform_noise.ndim == 3:
                    uniform_noise = uniform_noise[:, :, 0]
                randint_noise = uniform_noise * self.num_mask
                return randint_noise.clamp(0, self.num_mask - 1e-5).floor().long()
            self.rand_fn_reproducible = lambda *args, **kwargs: robust_seeded_randint(*args, **kwargs)
        elif self.noise_type == 'randn':
            self.rand_fn = lambda *args, **kwargs: torch.randn(*args, **kwargs) * self.noise_scale
            self.rand_fn_reproducible = lambda *args, **kwargs: robust_seeded_randn(*args, mode='randn', **kwargs) * self.noise_scale
        elif self.noise_type == 'learnable':
            self.rand_fn = lambda *args, **kwargs: (torch.randn(*args, **kwargs) * self.noise_scale) * self.backbone.vocab_embed.learnable_noise_sigma + self.backbone.vocab_embed.learnable_noise_mu
            self.rand_fn_reproducible = lambda *args, **kwargs: (robust_seeded_randn(*args, mode='randn', **kwargs) * self.noise_scale) * self.backbone.vocab_embed.learnable_noise_sigma + self.backbone.vocab_embed.learnable_noise_mu
        else:
            self.rand_fn = lambda *args, **kwargs: (torch.rand(*args, **kwargs) * 2 - 1.) * self.noise_scale
            self.rand_fn_reproducible = lambda *args, **kwargs: (robust_seeded_randn(*args, mode='rand', **kwargs) * 2 - 1.) * self.noise_scale
    
    def _process_model_output(self, model_output, xt, sigma):
        model_output, epsilon = model_output
        del sigma
        
        # SUBS masking
        mask_idx_tensor = torch.tensor([self.mask_index], device=model_output.device)
        model_output = model_output.index_fill(dim=-1, index=mask_idx_tensor, value=self.neg_infinity)
        
        if xt.ndim == 3:
            xt = xt.argmax(-1)
        
        model_output = model_output.log_softmax(dim=-1)
        
        # Carry Over
        unmasked_mask = (xt != self.mask_index).unsqueeze(-1)
        replacement = torch.full_like(model_output, self.neg_infinity)
        replacement = torch.scatter(replacement, -1, xt.unsqueeze(-1), torch.zeros_like(replacement[..., :1]))
        model_output = torch.where(unmasked_mask, replacement, model_output)
        return model_output, epsilon
    
    def _post_process_update(self, x, fix_eps=False):
        reset = x >= self.vocab_size
        x = torch.where(
            reset, torch.full_like(x, self.mask_index), x)
        mask = x == self.mask_index
        if fix_eps:
            reset = torch.zeros_like(reset)
        return x, reset, mask

    def _compute_posterior(self, x, xt, alpha_s, alpha_t):
        """Computes the posterior / approximate posterior.

        Args:
            x: Either clean input `x0` (one-hot),
                or model's predicted `x_theta` of shape (B, L, V).
            xt: The noisy latent (as indices) of shape (B, L).
            alpha_s: Noise level at s of shape (B, [L | 1], 1).
            alpha_t: Noise level at t of shape (B, [L | 1], 1).

        Returns:
            Posterior / approximate posterior of shape (B, L, V+1).
            Last_dim is for reset token.
        """
        if self.config.sampling.use_float64:
            x = x.to(torch.float64)
        if alpha_s.ndim == 2:
            alpha_s = alpha_s.unsqueeze(-1)
        if alpha_t.ndim == 2:
            alpha_t = alpha_t.unsqueeze(-1)
        alpha_ts = alpha_t / alpha_s
        d_alpha = alpha_s - alpha_t
        xt_one_hot = F.one_hot(xt, self.vocab_size).to(
            self.dtype).to(self.device)
        posterior = (alpha_ts - alpha_t) * xt_one_hot + d_alpha * x
        posterior = torch.cat([posterior, torch.zeros_like(
            posterior[:, :, :1])], dim=-1) # add reset token dim
        posterior[:, :, -1:] = (1 - alpha_ts) * (1 - alpha_s)
        posterior = posterior / (1 - alpha_t)
        return posterior
    
    @torch.no_grad()
    def _ancestral_update(self, x, t, dt, p_x0=None,
                          noise_removal_step=False,
                          epsilon=None, reset=None, mask=None):
        _, alpha_t = self.noise(t)
        if noise_removal_step:
            alpha_s = torch.ones_like(alpha_t)
        else:
            _, alpha_s = self.noise(t - dt)
        sigma_t = self._sigma_from_alphat(alpha_t)
        assert alpha_t.ndim == 2
        if p_x0 is None:
            x_prob, epsilon = self.forward(
                x, sigma_t,
                epsilon=epsilon, reset=reset, mask=mask)
            x_prob = x_prob.exp()
        else:
            x_prob = p_x0
         
        # Compute posterior / approximate posterior
        if self.config.sampling.mdlm_posterior:
            if self.config.sampling.use_p_nucleus and self.config.sampling.p_nucleus < 1.0:
                x_prob = utils.top_k_top_p_filtering(
                    x_prob.log(), top_p=self.config.sampling.p_nucleus).exp()
                x_prob = x_prob / x_prob.sum(dim=-1, keepdim=True)
            q_xs = x_prob * (alpha_s - alpha_t)[:, :, None]
            q_xs[:, :, self.mask_index]= 1 - alpha_s
            x_next = trainer_base.sample_categorical(q_xs, self.config.sampling.temperature)

            copy_flag = (x != self.mask_index).to(x.dtype)
            x_next = x_next * (1 - copy_flag) + x * copy_flag
        else:
            if self.config.sampling.use_p_nucleus and self.config.sampling.p_nucleus < 1.0:
                x_prob = utils.top_k_top_p_filtering(
                    x_prob.log(), top_p=self.config.sampling.p_nucleus).exp()
                x_prob = x_prob / x_prob.sum(dim=-1, keepdim=True)
            q_xs = self._compute_posterior(
                x=x_prob,
                xt=x,
                alpha_s=alpha_s,
                alpha_t=alpha_t
                )
            x_next = trainer_base.sample_categorical(q_xs, self.config.sampling.temperature)
            copy_flag = (x != self.mask_index).to(x.dtype)
            x_next = x_next * (1 - copy_flag) + x * copy_flag
        return x_prob, x_next, epsilon

    def restore_model_and_sample(self, num_steps, eps=1e-5, duplicate=0, num_duplicate_batch=1, xT=None, given_t=None):
        """Generate samples from the model."""
        if duplicate > 1:
            assert self.config.loader.eval_batch_size % num_duplicate_batch == 0
            assert duplicate % num_duplicate_batch == 0
            self._eval_mode()
            samples = []
            noise = self.prior_sample(self.config.loader.eval_batch_size // duplicate, self.num_tokens)
            noise = noise.repeat((duplicate // num_duplicate_batch), 1)
            seed = torch.randint(0, 1000000, (self.config.loader.eval_batch_size // duplicate, ), device=self.device)
            seed = seed.repeat((duplicate // num_duplicate_batch))
            for _ in range(num_duplicate_batch):
                sample = self.generate_samples(
                    num_samples=self.config.loader.eval_batch_size,
                    num_steps=num_steps,
                    eps=eps,
                    xT=noise,
                    given_t=given_t,
                    seed=seed)
                samples.append(sample)
            sample = torch.cat(samples, dim=0)
            print('Generated samples with duplicate:', sample.shape)
            self._train_mode()
            return sample
        
        self._eval_mode()
        samples = self.generate_samples(
            num_samples=self.config.loader.eval_batch_size,
            num_steps=num_steps,
            eps=eps,
            xT=xT,
            given_t=given_t)
        self._train_mode()
        return samples
    
    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None,
                                             eps=1e-5, xT=None, given_t=None, seed=None):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        fix_eps = self.config.sampling.get('fix_eps', False)
        noise_dim = self.config.model.get('noise_dim', self.config.model.hidden_size)
        if num_steps is None:
            num_steps = self.config.sampling.steps
        if xT is None:
            x = self.prior_sample(num_samples, self.num_tokens)
        else:
            x = xT
        if given_t is None:
            given_t = torch.ones(x.shape[0], device=self.device)
        B, N = x.shape
        D = noise_dim
        
        if seed is None:
            print('Using non-reproducible sampling')
            epsilon = self.rand_fn(B, N, D, device=self.device)
        else:
            print('Using reproducible sampling')
            epsilon = self.rand_fn_reproducible(
                N, D, seed=seed, device=self.device) # seed: B 1
            
        timesteps = torch.linspace(
            1.0, eps, num_steps + 1, device=self.device)
        timesteps = timesteps[:, None] * given_t[None, :] # [T, B]
        dt = (given_t[:, None] + eps) / num_steps
        p_x0_cache = None

        B, N = x.shape
        # D = self.config.model.hidden_size
        reset = torch.zeros(B, N, device=self.device)
        mask = x == self.mask_index

        cache_count = 0
        for i in range(num_steps):
            t = timesteps[i].unsqueeze(1) # [B, 1] # * torch.ones(x.shape[0], 1, device=self.device)
            is_last_step = (i == num_steps - 1)

            if is_last_step:
                if self.config.sampling.noise_removal == 'greedy':
                    sigma = self._sigma_from_alphat(self.noise(t)[1])
                    x, _ = self.forward(xt=x, sigma=sigma, epsilon=epsilon, reset=reset, mask=mask)
                    x = x.argmax(dim=-1)
                else:
                    if self.sampler == 'analytic':
                        raise NotImplementedError
                    else:
                        _, x, epsilon = self._ancestral_update(x=x, t=t, dt=None,
                            p_x0=p_x0_cache,
                            epsilon=epsilon, reset=reset, mask=mask,
                            noise_removal_step=True)

            elif self.sampler == 'ancestral':
                _, x, epsilon = self._ancestral_update(
                    x=x, t=t, dt=dt, p_x0=None,
                    epsilon=epsilon, reset=reset, mask=mask
                    )
            elif self.sampler == 'ancestral_cache':
                p_x0_cache, x_next, epsilon = self._ancestral_update(
                    x=x, t=t, dt=dt, p_x0=p_x0_cache,
                    epsilon=epsilon, reset=reset, mask=mask
                    )
                if (not torch.allclose(x_next, x)) or self.time_conditioning:
                    p_x0_cache = None
                else:
                    cache_count += 1
                x = x_next
            else:
                raise NotImplementedError
            x, reset, mask = self._post_process_update(x, fix_eps=fix_eps)
        
        return x
    
    def restore_model_and_cond_sample(self, num_steps, data, prefix_len=50, eps=1e-5, duplicate=0, xT=None, given_t=None):        
        self._eval_mode()
        samples = self.generate_cond_samples(
            num_samples=self.config.loader.eval_batch_size,
            data=data,
            prefix_len=prefix_len,
            num_steps=num_steps,
            eps=eps,
            xT=xT,
            given_t=given_t)
        self._train_mode()
        return samples
    
    @torch.no_grad()
    def generate_cond_samples(self, num_samples, data, prefix_len=50, 
                               num_steps=None, eps=1e-5, xT=None, given_t=None, seed=None):
        """Generate samples from the model."""
        # build xT and pass to generate_samples
        # Lightning auto-casting is not working in this method for some reason
        xT = data.clone()
        unfix_len = (xT.shape[1] - prefix_len)
        xT[:, -unfix_len:] = -1
        fillin_mask = xT == -1
        xT = torch.where(fillin_mask, self.mask_index, xT)
        x = self.generate_samples(
            num_samples=num_samples,
            num_steps=num_steps,
            eps=eps,
            xT=xT,
            given_t=given_t,
            seed=seed,
            )
        x = torch.where(fillin_mask, x, data)
        return x
    
    def nll(self, x0, output_tokens,
                    current_accumulation_step=None, train_mode=False, xT=None, **kwargs):
        del output_tokens
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        assert t.shape[0] == x0.shape[0]
        
        dalpha_t, alpha_t = self.noise(t)
        
        if train_mode:
            alpha_t = self.alpha_min + alpha_t * (self.alpha_max - self.alpha_min)
            dalpha_t = dalpha_t * (self.alpha_max - self.alpha_min)
        alpha_t = alpha_t.unsqueeze(-1)
        dalpha_t = dalpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(alpha_t)

        xt = self.q_xt(x0, alpha_t)
        mask = xt == self.mask_index
        epsilon = None
        reset = None
        log_x_theta, _ = self.forward(xt, sigma=sigma, epsilon=epsilon, reset=reset, mask=mask)
        utils.print_nans(log_x_theta, 'model_output')
        return self.nll_per_token(
            log_x_theta=log_x_theta,
            xt=xt,
            x0=x0,
            alpha_t=alpha_t,
            dalpha_t=dalpha_t,
            low_var=train_mode and self.loss_type == 'low_var')

class IMReDi(IMDM): # Training as duo, without curriculum
    def nll(self, x0, output_tokens,
                    current_accumulation_step=None, train_mode=False, xT=None, given_t=None, not_sampling_t=False,
                    seed=None):
        if not train_mode:
            return super().nll(x0, output_tokens, current_accumulation_step, train_mode, xT)
        del output_tokens
        xt = xT        
        t = given_t
        dalpha_t, alpha_t = self.noise(t)
        noise_dim = self.config.model.get('noise_dim', self.config.model.hidden_size)
        
        alpha_t = alpha_t.unsqueeze(-1)
        dalpha_t = dalpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(alpha_t)

        assert seed is not None, 'Rectification requires reproducible noise'
        B, N = xt.shape
        D = noise_dim
        epsilon = self.rand_fn_reproducible(
            N, D, seed=seed, device=self.device) # seed: B 1
        if self.config.algo.get('ignore_seed', False):
            epsilon = self.rand_fn(B, N, D, device=self.device)
        reset = torch.zeros_like(xt) # Keep epsilon for ODE path.
        mask = xt == self.mask_index
        log_x_theta, _ = self.forward(xt, sigma=sigma, epsilon=epsilon, reset=reset, mask=mask)

        return self.nll_per_token(log_x_theta=log_x_theta,
                                    xt=xt,
                                    x0=x0,
                                    alpha_t=alpha_t,
                                    dalpha_t=dalpha_t,
                                    low_var=train_mode and self.loss_type == 'low_var',
                                    simple_loss=False,
                                    )

class IMSDTT(IMDM):
    def __init__(self, config, tokenizer, verbose=True):
        super().__init__(config, tokenizer)
        self.neg_infinity = -1000000.0
        self.verbose = verbose

        self.teacher = None
        self.num_distill_steps = self.config.algo.num_distill_steps
        self.tot_num_sampl_steps = self.config.algo.orig_num_sampling_steps
        self.min_num_sampl_steps = self.config.algo.min_num_sampling_steps
        self.distill_mode = self.config.algo.distill_mode
        self.reset_optimizer_on_growth = (
            config.algo.reset_optimizer_on_growth
        )
        self.use_ema_on_growth = config.algo.use_ema_on_growth

        self.sampling_eps_tensor = torch.tensor(self.sampling_eps)
        self.sampling_mode = self.config.sampling.predictor
        assert self.sampling_mode in ("ancestral", "analytic", "ancestral_cache")

        self.grow_dt_every = config.algo.grow_dt_every

        self.dt = (1 - self.sampling_eps) / self.tot_num_sampl_steps
        self.loss_precision = self.config.training.loss_precision

        mode = self.distill_mode
        self._loss_fn = None  # fn to compare preds & targets
        if mode == "mse":
            self._loss_fn = self._mse
        elif mode == "tvd":
            self._loss_fn = self._tvd
        elif mode == "kl-fwd":
            self._loss_fn = self._fwd_kl
        elif mode == "kl-bwd":
            self._loss_fn = self._bwd_kl
        else:
            raise ValueError(mode)
        # self.prepare_teacher_and_student() # will call after loading the checkpoint outside of algo.py (see main.py)


    def prepare_teacher_and_student(self):
        """
        If start from hf checkpoint:
            - Load the hf arch in student + teacher
        Else:
            - Init teacher as a copy of student

        if start checkpoint is not kuleshov-group/mdlm-owt -> load from disk

        """
        # Hack so that teacher doesn't get registered as child
        self.teacher = [
            copy.deepcopy(self.backbone).eval()
        ]

        self.teacher[0].eval()
        self.teacher[0].requires_grad_(False)
        # see main.py for ema load
        # self.init_ema()

    def forward_teacher(self, xt, sigma, **kwargs):
        sigma = self._process_sigma(sigma)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            model_output = self.teacher[0](xt, sigma, **kwargs)
        return self._process_model_output(
            model_output=model_output, xt=xt, sigma=sigma)

    def to(self, device):
        super().to(device=device)
        self.teacher[0].to(device=device)

    @torch.no_grad
    def _teacher_logprobs_on_mask(self, xt, t_start):
        """
        Collect teacher predictions for ALL mask tokens
        """
        dt = self.dt
        noise_dim = self.config.model.get('noise_dim', self.config.model.hidden_size)

        space = torch.linspace(
            1, 0, self.num_distill_steps, device=t_start.device
        ).double()[:, None]
        t_start = t_start[None, :].double()
        t_end = t_start - dt * self.num_distill_steps
        # Evenly-spaced interpolation between t_start and t_end
        ts = t_start * space + (1 - space) * t_end
        # Ensure we don't feed the model values smaller than sampling_eps
        ts = torch.maximum(ts, self.sampling_eps_tensor)

        teacher_predictions = torch.zeros(
            (*xt.shape, self.vocab_size), device=xt.device
        )
        unmasked_tokens = torch.zeros(xt.shape, device=xt.device)
        curr_x = xt
        
        B, N = xt.shape
        D = noise_dim
        epsilon = self.rand_fn(B, N, D, device=xt.device)
        reset = torch.zeros_like(curr_x).bool()
        fix_eps = self.config.algo.get('sdtt_fix_eps', True) # default True
        for idx in range(len(ts)):
            t = ts[idx].float()
            # TODO: add analytic sampler
            if self.sampling_mode in ["ancestral", "ancestral_cache"]:
                alpha_t = self.noise(t)[1].unsqueeze(-1)
                alpha_s = self.noise(t - dt)[1].unsqueeze(-1)
                sigma_t = self._sigma_from_alphat(alpha_t)
                mask = curr_x == self.mask_index
                log_p_x0, epsilon = self.forward_teacher(
                        curr_x, sigma_t,
                        epsilon=epsilon, reset=reset, mask=mask)
                x_prob = log_p_x0.exp()
                q_xs = self._compute_posterior(
                    x=x_prob,
                    xt=curr_x,
                    alpha_s=alpha_s,
                    alpha_t=alpha_t
                    )
                update = trainer_base.sample_categorical(q_xs, precision="float64") # do not follow legacy
                copy_flag = (curr_x != self.mask_index).to(curr_x.dtype)
                updated = update * (1 - copy_flag) + curr_x * copy_flag
                new_batch, reset, mask = self._post_process_update(updated, fix_eps=fix_eps)

            elif self.sampling_mode == "analytic":
                raise NotImplementedError
            else:
                raise ValueError(self.sampling_mode)

            updated = curr_x != new_batch
            # Extract predictions for denoised tokens
            teacher_predictions[updated] = log_p_x0[updated].to(teacher_predictions.dtype)
            unmasked_tokens += updated
            curr_x = new_batch

        # Put predictions from model on last step for remaining MASK tokens
        last_preds_update_mask = (curr_x == self.mask_index) * torch.logical_not(
            unmasked_tokens
        )
        last_preds_update_mask = last_preds_update_mask[..., None].to(bool)
        teacher_predictions = torch.where(
            last_preds_update_mask, log_p_x0.to(teacher_predictions.dtype), teacher_predictions
        )
        return teacher_predictions, epsilon

    def nll(self, x0, output_tokens,
                    current_accumulation_step=None, train_mode=False, xT=None, **kwargs):
        del output_tokens
        t = self._sample_t(x0.shape[0], current_accumulation_step)

        dalpha_t, alpha_t = self.noise(t)
        alpha_t = alpha_t.unsqueeze(-1)
        dalpha_t = dalpha_t.unsqueeze(-1)
        xt = self.q_xt(x0, alpha_t)
        
        sigma = self._sigma_from_alphat(alpha_t)
        # Loss on all masked tokens
        teacher_preds, epsilon = self._teacher_logprobs_on_mask(xt, t)
        
        mask = xt == self.mask_index
        reset = torch.zeros_like(xt).bool()        
        student_preds, _ = self.forward(xt, sigma, epsilon=epsilon, reset=reset, mask=mask)
        is_mask = xt == self.mask_index

        target = teacher_preds
        preds = student_preds

        if self.loss_precision == "64":
            target = target.to(torch.float64)
            preds = preds.to(torch.float64)
        elif self.loss_precision == "32":
            target = target.to(torch.float32)
            preds = preds.to(torch.float32)

        loss = self._loss_fn(preds, target)
        loss = loss[is_mask]
        return loss

    def _mse(self, preds, target):
        return F.mse_loss(preds, target, reduction="none").sum(-1)

    def _tvd(self, preds, target):
        return (preds - target).abs().sum(-1)

    def _fwd_kl(self, preds, target):
        return F.kl_div(preds, target, log_target=True, reduction="none").sum(-1)

    def _bwd_kl(self, preds, target):
        return F.kl_div(target, preds, log_target=True, reduction="none").sum(-1)

    def training_step(self, batch, batch_idx):
        step = self.trainer.global_step
        curr_round = step // self.grow_dt_every
        self.dt = (
            (1 - self.sampling_eps) / self.tot_num_sampl_steps * (self.num_distill_steps**curr_round)
        )

        if step > 0 and step % self.grow_dt_every == 0:
            curr_round = step // self.grow_dt_every
            self.dt = (
                (1 - self.sampling_eps) / self.tot_num_sampl_steps * (self.num_distill_steps**curr_round)
            )
            effective_num_steps = round(1 / self.dt)
            if effective_num_steps < self.min_num_sampl_steps:
                print(
                    f"Reached below the minimal effective number of sampling steps, stopping..."
                )
                import sys
                sys.exit()
            else:
                print(
                    f"Step {step}: Doubling `dt`! New effective number of steps: {effective_num_steps}."
                )
            self._student_to_teacher()
            if self.reset_optimizer_on_growth:
                print("Resetting optimizers...")
                self.trainer.strategy.setup_optimizers(self.trainer)
        return super().training_step(batch, batch_idx)

    def _student_to_teacher(
        self,
    ):
        start = time.perf_counter()
        if self.use_ema_on_growth:
            # Use EMA as teacher and student, and reset EMA for next round
            self.store_ema()
            student_ckpt = copy.deepcopy(self.backbone.state_dict())
            self.restore_ema()
            self.backbone.load_state_dict(student_ckpt)
            self.init_ema()
        else:
            student_ckpt = self.backbone.state_dict()

        self.teacher[0].load_state_dict(student_ckpt)
        end = time.perf_counter()

        print(f"Swapped student into teacher in {end - start:.2f} seconds.")
        # get save_path of checkpoint callback
        save_path = self.trainer.checkpoint_callback.dirpath
        save_path = os.path.join(save_path, "student_checkpoints", f"{self.trainer.global_step}.ckpt")
        self.trainer.save_checkpoint(save_path)
