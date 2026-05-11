import itertools
import os

from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers

import dataloader
import metrics
import models
import utils
from omegaconf import ListConfig


@dataclass
class Loss:
    loss: torch.FloatTensor
    nlls: torch.FloatTensor
    prior_loss: torch.FloatTensor
    num_tokens: torch.FloatTensor


class LogLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-3  # To be consistent with SEDD: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/noise_lib.py#L56

    def forward(self, t):
        t = (1 - self.eps) * t
        alpha_t = 1 - t 
        dalpha_t = - (1 - self.eps) + t * 0
        assert alpha_t.shape == dalpha_t.shape
        return dalpha_t, alpha_t


def sample_categorical(categorical_probs, temperature=1.0, precision="float64"):
    if '64' in str(precision):
        categorical_probs = categorical_probs.to(torch.float64)
    if temperature != 1.0:
        categorical_probs = categorical_probs.pow(1.0 / temperature)
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
    return x.view(
        * x.shape,
        * ((1,) * (len(reference.shape) - len(x.shape))))


class TrainerBase(L.LightningModule):
    def __init__(
        self,
        config,
        tokenizer: transformers.PreTrainedTokenizer,
        weights_only=False,
        vocab_size=None):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        if hasattr(self.config.algo, 'ignore_bos'):
            self.ignore_bos = config.algo.ignore_bos
        else:
            self.ignore_bos = False
        if hasattr(self.config.algo, 'loss_type'):
            self.loss_type = config.algo.loss_type
        self.tokenizer = tokenizer
        if vocab_size is None:
            self.vocab_size = len(self.tokenizer)
        else:
            self.vocab_size = vocab_size
        self.sampler = self.config.sampling.predictor
        self.antithetic_sampling = self.config.training.antithetic_sampling
        self.parameterization = self.config.algo.parameterization
        if self.config.algo.backbone == 'dit':
            self.backbone = models.dit.DIT(
                self.config, vocab_size=self.vocab_size)
        elif self.config.algo.backbone == 'dimamba':
            self.backbone = models.dimamba.DiMamba(
                self.config,
                vocab_size=self.vocab_size,
                pad_token_id=self.tokenizer.pad_token_id)
        elif self.config.algo.backbone == 'hf_dit':
            self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
                config.eval.checkpoint_path, trust_remote_code=True)

        self.T = self.config.algo.T
        self.num_tokens = self.config.model.length
        self.softplus = torch.nn.Softplus()
        self.p_nucleus = self.config.sampling.p_nucleus
        # Noise Schedule
        self.noise = LogLinear()

        self.metrics = metrics.Metrics(
            gen_ppl_eval_model_name_or_path=\
                self.config.eval.gen_ppl_eval_model_name_or_path,
            eval_ppl_batch_size=\
                self.config.eval.perplexity_batch_size)

        if self.config.training.ema > 0:
            self.ema = models.ema.ExponentialMovingAverage(
                self._get_parameters(),
                decay=self.config.training.ema)
        else:
            self.ema = None
        
        self.lr = self.config.optim.lr
        self.sampling_eps = self.config.training.sampling_eps
        self.time_conditioning = self.config.algo.time_conditioning
        self.neg_infinity = -1000000.0

    def _validate_configuration(self):
        assert self.config.algo.backbone in {'dit', 'hf_dit'}
        if self.config.algo.parameterization == 'ar':
            assert not self.config.algo.time_conditioning
            assert self.config.prior.type == 'none'

        if self.parameterization in {'score', 'mean'}:
            assert self.time_conditioning
        if self.T > 0:
            assert self.parameterization != 'score'
        # assert not self.config.algo.time_conditioning # Disabled for now

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.metrics.to(*args, **kwargs)
        return self

    def q_xt(self, x, alpha_t):
        raise NotImplementedError

    def _get_parameters(self):
        return itertools.chain(self.backbone.parameters(),
                                                     self.noise.parameters())
    
    def _get_named_parameters(self):
        return itertools.chain(self.backbone.named_parameters(),
                                                     self.noise.named_parameters())

    def _eval_mode(self):
        if self.ema and not self.config.eval.disable_ema:
            print('Copying EMA parameters to model')
            self.ema.store(self._get_parameters())
            self.ema.copy_to(self._get_parameters())
        else:
            print('No EMA parameters')
        self.backbone.eval()
        self.noise.eval()

    def _train_mode(self):
        if self.ema and not self.config.eval.disable_ema:
            self.ema.restore(self._get_parameters())
        self.backbone.train()
        self.noise.train()

    def on_load_checkpoint(self, checkpoint):
        if len(self.state_dict()) != len(checkpoint['state_dict']):
            print('Parameter count mismatch between model and checkpoint.')
            print(f'Model parameters: {len(list(self._get_parameters()))}')
            print(f'Checkpoint parameters: {len(checkpoint["state_dict"])}')

            # TODO: match parameters by name.            
            assert len(self.noise.state_dict()) == 0
            model_state_dict = self.state_dict()
            loaded_state_dict = checkpoint['state_dict']
            
            del model_state_dict['backbone.rotary_emb.inv_freq']
            del loaded_state_dict['backbone.rotary_emb.inv_freq']
            
            new_shadow_params = []
            loaded_ckpt_key_to_idx = {k: idx for idx, k in enumerate(loaded_state_dict.keys())}
            for k, v in model_state_dict.items():
                if k in loaded_state_dict:
                    new_shadow_params.append(checkpoint['ema']['shadow_params'][loaded_ckpt_key_to_idx[k]])
                else:
                    new_shadow_params.append(v.clone().detach())
            checkpoint['ema']['shadow_params'] = new_shadow_params
        
        if self.ema:
            self.ema.load_state_dict(checkpoint['ema'])

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()

    def on_train_start(self):
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(self._get_parameters())

    def _process_sigma(self, sigma):
        raise NotImplementedError

    def _process_model_output(self, model_output, xt, sigma):
        raise NotImplementedError

    def forward(self, xt, sigma, **kwargs):
        sigma = self._process_sigma(sigma)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            model_output = self.backbone(xt, sigma, **kwargs)
        return self._process_model_output(
            model_output=model_output, xt=xt, sigma=sigma)

    def on_train_epoch_start(self):
        self.metrics.reset()
        assert self.metrics.train_nlls.nll.mean_value == 0
        assert self.metrics.train_nlls.nll.weight == 0

    def training_step(self, batch, batch_idx):
        current_accumulation_step = (
            batch_idx % self.trainer.accumulate_grad_batches)
        kwargs = {}
        for k, v in batch.items():
            if k not in {'input_ids', 'attention_mask'}:
                kwargs[k] = v
        losses = self._loss(batch['input_ids'],
                                                batch['attention_mask'],
                                                current_accumulation_step,
                                                train_mode=True,
                                                **kwargs
                                                )
        self.metrics.update_train(losses.nlls, losses.prior_loss,
                                                            losses.num_tokens)
        self.log(name='trainer/loss',
                         value=losses.loss.item(),
                         on_step=True,
                         on_epoch=False,
                         prog_bar=True,
                         sync_dist=True)
        return losses.loss

    def on_before_optimizer_step(self, optimizer):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float('inf'))
        self.log("train/grad_norm", grad_norm, prog_bar=True, logger=True)

        # final_layer grad norm
        final_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.backbone.output_layer.linear.parameters(), max_norm=float('inf'))
        self.log("train/final_grad_norm", final_grad_norm, prog_bar=False, logger=True)

        # IMDM MLP last layer weight norm
        if self.backbone.is_imdm:
            if not self.as_mdlm and self.num_mask < 0:
                imdm_weight = self.backbone.vocab_embed.imdm_mlp[-1].weight.data
                imdm_weight_norm = imdm_weight.norm()
                self.log("train/imdm_weight_norm", imdm_weight_norm, prog_bar=False, logger=True)

    def on_train_epoch_end(self):
        for k, v in self.metrics.valid_nlls.items():
            self.log(name=k, value=v.compute(), on_step=False,
                             on_epoch=True, sync_dist=True)

    def on_validation_epoch_start(self):
        self.metrics.reset()
        self._eval_mode()
        assert self.metrics.valid_nlls.nll.mean_value == 0
        assert self.metrics.valid_nlls.nll.weight == 0

    def validation_step(self, batch, batch_idx):
        del batch_idx
        kwargs = {}
        for k, v in batch.items():
            if k not in {'input_ids', 'attention_mask'}:
                kwargs[k] = v
        losses = self._loss(batch['input_ids'],
                            batch['attention_mask'],
                            **kwargs
                            )
        self.metrics.update_valid(losses.nlls, losses.prior_loss,
                                                            losses.num_tokens)
        return losses.loss

    def on_validation_epoch_end(self):
        for k, v in self.metrics.valid_nlls.items():
            self.log(name=k,  value=v.compute(), on_step=False,
                             on_epoch=True, sync_dist=True)
        if ((self.config.eval.compute_perplexity_on_sanity
                 or not self.trainer.sanity_checking)
                 and self.config.eval.generate_samples):
            
            if isinstance(self.config.sampling.steps, int):
                step_list = [self.config.sampling.steps]
            else:
                step_list = self.config.sampling.steps.split(',')
                step_list = [int(s.strip()) for s in step_list if s.strip()]
                
            for num_steps in step_list:
                if hasattr(self.metrics, 'gen_ppl'):
                    self.metrics.gen_ppl.reset()
                if hasattr(self.metrics, 'sample_entropy'):
                    self.metrics.sample_entropy.reset()

                current_text_samples = []

                for _ in range(self.config.sampling.num_sample_batches):
                    samples = self.generate_samples(
                        num_samples=self.config.loader.eval_batch_size,
                        num_steps=num_steps
                    )

                    self.metrics.record_entropy(samples)

                    decoded_batch = self.tokenizer.batch_decode(samples)

                    if len(current_text_samples) < self.config.sampling.num_sample_log:
                        current_text_samples.extend(decoded_batch)

                    if self.config.eval.compute_generative_perplexity:
                        self.metrics.record_generative_perplexity(
                            decoded_batch, self.num_tokens, self.device)

                if self.config.eval.compute_generative_perplexity:
                    self.log(f'val/gen_ppl_T{num_steps}',
                                self.metrics.gen_ppl.compute(),
                                on_epoch=True,
                                on_step=False,
                                sync_dist=True)
                    self.log(f'val/sample_entropy_T{num_steps}',
                                self.metrics.sample_entropy.compute(),
                                on_epoch=True,
                                on_step=False,
                                sync_dist=True)

                if self.trainer.global_rank == 0 and hasattr(self.trainer.logger, 'log_table'):
                    log_samples = current_text_samples[:self.config.sampling.num_sample_log]

                    self.trainer.logger.log_table(
                        key=f'samples_T{num_steps}@global_step{self.global_step}',
                        columns=['Generated Samples'],
                        data=[[s] for s in log_samples]
                    )

        self._train_mode()

    def on_test_epoch_start(self):
        self._eval_mode()
        self.xTx0s = []
        self.ts = []

    def test_step(self, batch, batch_idx):
        x0 = batch['input_ids']
        # t = torch.ones(x0.shape[0], device=self.device) * np.random.uniform(0, 1)
        t = torch.rand(x0.shape[0], device=self.device)
        if not self.config.reflow.perturbed_rect:
            t = torch.ones_like(t)
        xt = self.q_xt(x0, alpha_t=1-t.unsqueeze(-1))
        if self.config.reflow.get_seed:
            seed = torch.randint(0, 1000000, (x0.shape[0],), device=self.device)
        x0 = self.generate_samples(xt.shape[0], xT=xt.detach().clone(), given_t=t, seed=seed)
        if seed is not None:
            t = torch.stack([t, seed], dim=0) # 2 B
        else:
            t = t.unsqueeze(0) # 1 B
        pair = torch.stack([xt, x0], dim=0) # 2 B N
        self.xTx0s.append(pair)
        self.ts.append(t)
        return 0.

    def on_test_epoch_end(self):
        # gather across all GPUs
        self.xTx0s = torch.cat(self.xTx0s, dim=1) # 2 B N
        self.ts = torch.cat(self.ts, dim=1) # T B
        torch.distributed.barrier()

        # if multi gpu
        if torch.distributed.is_initialized():
            data_xTx0s_all = [torch.empty_like(self.xTx0s) for _ in range(
                    torch.distributed.get_world_size())] if self.trainer.global_rank == 0 else None
            torch.distributed.gather(self.xTx0s,
                                                                data_xTx0s_all,
                                                                dst=0)
            ts_all = [torch.empty_like(self.ts) for _ in range(
                    torch.distributed.get_world_size())] if self.trainer.global_rank == 0 else None
            torch.distributed.gather(self.ts,
                                                                ts_all,
                                                                dst=0)
                
        if self.trainer.global_rank == 0:
            xTx0s = torch.cat(data_xTx0s_all, dim=1).cpu()[:, :self.config.reflow.num_reflow_samples]
            xTs, x0s = xTx0s[0], xTx0s[1]
            ts = torch.cat(ts_all, dim=1).cpu()[:, :self.config.reflow.num_reflow_samples]
            if ts.shape[0] == 2:
                ts, seeds = ts[0], ts[1]
            else:
                ts = ts[0]
                seeds = None

            save_path = self.config.reflow.save_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            xTs = xTs.cpu().numpy()
            x0s = x0s.cpu().numpy()
            ts = ts.cpu().numpy()
            xT_path = os.path.join(save_path, 'xT.npy')
            x0_path = os.path.join(save_path, 'x0.npy')
            ts_path = os.path.join(save_path, 'ts.npy')
            
            if seeds is not None:
                seeds = seeds.long().cpu().numpy()
                seeds_path = os.path.join(save_path, 'seeds.npy')
                np.save(seeds_path, seeds)
                print('seeds shape:', seeds.shape)
                print('seeds saved to:', seeds_path)
            
            np.save(xT_path, xTs)
            np.save(x0_path, x0s)
            np.save(ts_path, ts)
            print('xT shape:', xTs.shape)
            print('x0 shape:', x0s.shape)
            print('ts shape:', ts.shape)
            print('xT saved to:', xT_path)
            print('x0 saved to:', x0_path)
            print('ts saved to:', ts_path)
        return

    def configure_optimizers(self):
        # TODO: filter params.
        if self.config.optim.ln_tune == 'norm2':
            params = []
            for name, param in self.named_parameters():
                if 'norm2.weight' in name:
                    print(name)
                    params.append(param)
            optimizer = torch.optim.AdamW(
                params,
                lr=self.config.optim.lr,
                betas=(self.config.optim.beta1,
                            self.config.optim.beta2),
                eps=self.config.optim.eps,
                weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.ln_tune == 'norm':
            params = []
            for name, param in self.named_parameters():
                if 'norm1.weight' in name:
                    print(name)
                    params.append(param)
                if 'norm2.weight' in name:
                    print(name)
                    params.append(param)
            optimizer = torch.optim.AdamW(
                params,
                lr=self.config.optim.lr,
                betas=(self.config.optim.beta1,
                            self.config.optim.beta2),
                eps=self.config.optim.eps,
                weight_decay=self.config.optim.weight_decay)
        else:
            optimizer = torch.optim.AdamW(
                self._get_parameters(),
                lr=self.config.optim.lr,
                betas=(self.config.optim.beta1,
                            self.config.optim.beta2),
                eps=self.config.optim.eps,
                weight_decay=self.config.optim.weight_decay)
        scheduler = hydra.utils.instantiate(
            self.config.lr_scheduler, optimizer=optimizer)
        scheduler_dict = {'scheduler': scheduler,
                                            'interval': 'step',
                                            'monitor': 'val/loss',
                                            'name': 'trainer/lr'}
        return [optimizer], [scheduler_dict]

    def generate_samples(self, num_samples, num_steps, eps, xT, given_t, **kwargs):
        raise NotImplementedError

    def restore_model_and_sample(self, num_steps, eps=1e-5, duplicate=0, num_duplicate_batch=1, xT=None, given_t=None):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        # for evaluating the tc
        if duplicate > 1:
            assert self.config.loader.eval_batch_size % num_duplicate_batch == 0
            assert duplicate % num_duplicate_batch == 0
            self._eval_mode()
            samples = []
            noise = self.prior_sample(self.config.loader.eval_batch_size // duplicate, self.num_tokens)
            seed = torch.randint(0, 1000000, (self.config.loader.eval_batch_size // duplicate, ), device=self.device)
            noise = noise.repeat((duplicate // num_duplicate_batch), 1)
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

    def _process_model_input(self, x0, valid_tokens):
        raise NotImplementedError

    def nll(self, input_tokens, output_tokens,
                    current_accumulation_step=None, train_mode=False, **kwargs):
        raise NotImplementedError

    def _loss(self, x0, valid_tokens,
                        current_accumulation_step=None,
                        train_mode=False,
                        xT=None, given_t=None, not_sampling_t=False, **kwargs):
        (input_tokens, output_tokens,
         valid_tokens) = self._process_model_input(
             x0, valid_tokens)
        loss = self.nll(input_tokens, output_tokens,
                                        current_accumulation_step, train_mode, xT=xT, given_t=given_t,
                                        not_sampling_t=not_sampling_t, **kwargs)
        if loss.ndim == 2:
            if self.ignore_bos:
                loss[:, 1:] = loss[:, 1:]
                valid_tokens[:, 1:] = valid_tokens[:, 1:]

            nlls = (loss * valid_tokens).sum()
            num_tokens = valid_tokens.sum()
            token_nll = nlls / num_tokens

            return Loss(loss=token_nll,
                                    nlls=nlls,
                                    prior_loss=0.0,
                                    num_tokens=num_tokens)
        elif loss.ndim == 1:
            nlls = loss.sum()
            num_tokens = loss.numel()
            token_nll = nlls / num_tokens
            
            return Loss(loss=token_nll,
                                    nlls=nlls,
                                    prior_loss=0.0,
                                    num_tokens=num_tokens)


class Diffusion(TrainerBase):
    def _validate_configuration(self):
        super()._validate_configuration()
        assert self.config.sampling.noise_removal in {
            'none', 'ancestral', 'greedy'}
        assert self.loss_type in {'elbo', 'low_var'}
        if self.config.sampling.noise_removal == 'greedy':
            assert self.sampler != 'analytic'
            assert self.parameterization in {'mean', 'subs'}

    def _process_model_input(self, x0, valid_tokens):
        return x0, None, valid_tokens

    def _process_sigma(self, sigma):
        assert sigma.ndim == 2
        sigma = sigma.mean(-1).squeeze()
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _sample_t(self, n, accum_step, given_t=None):
        if accum_step is not None:
            # During training
            batch_dim = n
            n = self.config.loader.global_batch_size
        _eps_t = torch.rand(n, device=self.device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=self.device) / n
            _eps_t = (_eps_t / n + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        if accum_step is not None:
            t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
            t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
            t = t.chunk(self.trainer.accumulate_grad_batches)[
                accum_step]
            # corner case for the last datapoint
            t = t[:batch_dim]
        if given_t is not None:
            if torch.is_tensor(given_t):
                assert given_t.shape == t.shape or given_t.numel() == 1, \
                        f"Shape mismatch: t={t.shape}, given_t={given_t.shape}"
            t = t * given_t
        return t

    def _sigma_from_alphat(self, alpha_t):
        return -torch.log(alpha_t)

    def _reconstruction_loss(self, x0):
        t0 = torch.zeros(1, x0.shape[0], dtype=self.dtype,
                                         device=self.device)
        sigma_t0 = self._sigma_from_alphat(self.noise(t0)[1])
        model_output_t0 = self.forward(x0, sigma_t0)
        return - torch.gather(input=model_output_t0,
                                                    dim=-1,
                                                    index=x0[:, :, None]).squeeze(-1)


    def nll_per_token(self, model_output, xt, x0, alpha_t,
                                        dalpha_t, low_var):
        raise NotImplementedError

    def nll(self, x0, output_tokens,
                    current_accumulation_step=None, train_mode=False, **kwargs):
        del output_tokens
        t = self._sample_t(x0.shape[0],
                                             current_accumulation_step)
        assert t.shape[0] == x0.shape[0]
        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            # t \in {1/T, 2/T, ..., 1}
            t += (1 / self.T)
        
        dalpha_t, alpha_t = self.noise(t)
        alpha_t = alpha_t.unsqueeze(-1)
        dalpha_t = dalpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(alpha_t)

        xt = self.q_xt(x0, alpha_t)
        log_x_theta = self.forward(xt, sigma=sigma)
        utils.print_nans(log_x_theta, 'model_output')
        return self.nll_per_token(
            log_x_theta=log_x_theta,
            xt=xt,
            x0=x0,
            alpha_t=alpha_t,
            dalpha_t=dalpha_t,
            low_var=train_mode and self.loss_type == 'low_var')

    def _get_score(self, **kwargs):
        del kwargs
        raise NotImplementedError

    def _denoiser_update(self, x, t):
        raise NotImplementedError

    def _analytic_update(self, x, t, dt):
        raise NotImplementedError

    def _ancestral_update(self, x, t, dt, p_x0, noise_removal_step):
        raise NotImplementedError

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None,
                                             eps=1e-5, xT=None, given_t=None, **kwargs):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        is_conditional = self.config.sampling.conditional
        if num_steps is None:
            num_steps = self.config.sampling.steps
            if isinstance(num_steps, str):
                num_steps = num_steps.split(',')
                num_steps = [int(s.strip()) for s in num_steps if s.strip()]
                assert len(num_steps) == 1, "During sampling, num_steps should be a single integer."
                num_steps = num_steps[0]
            else:
                num_steps = int(num_steps)
        if xT is None:
            x = self.prior_sample(num_samples, self.num_tokens)
        else:
            x = xT
        if given_t is None:
            given_t = torch.ones(x.shape[0], device=self.device)
        
        B, N = x.shape
        timesteps = torch.linspace(
            1.0, eps, num_steps + 1, device=self.device)
        timesteps = timesteps[:, None] * given_t[None, :] # [T, B]
        dt = (given_t[:, None] + eps) / num_steps
        p_x0_cache = None

        cache_count = 0
        if is_conditional:
            condition = xT > -1
            xT_prior = self.prior_sample(B, N)
            x = torch.where(condition, xT, xT_prior) # randomize unconditioned positions
            
        for i in range(num_steps):
            t = timesteps[i].unsqueeze(1) # [B, 1] # * torch.ones(x.shape[0], 1, device=self.device)
            is_last_step = (i == num_steps - 1)

            if is_last_step:
                if self.config.sampling.noise_removal == 'greedy':
                    sigma = self._sigma_from_alphat(self.noise(t)[1])
                    x = self.forward(xt=x, sigma=sigma).argmax(dim=-1)
                else:
                    if self.sampler == 'analytic':
                        x = self._denoiser_update(x=x, t=t)
                    else:
                        _, x = self._ancestral_update(x=x, t=t, dt=None,
                                                                        p_x0=p_x0_cache,
                                                                        noise_removal_step=True)
            elif self.sampler == 'ancestral':
                _, x = self._ancestral_update(
                    x=x, t=t, dt=dt, p_x0=None)
            elif self.sampler == 'ancestral_cache':
                p_x0_cache, x_next = self._ancestral_update(
                    x=x, t=t, dt=dt, p_x0=p_x0_cache)
                if (not torch.allclose(x_next, x)) or self.time_conditioning:
                    # Disable caching
                    p_x0_cache = None
                else:
                    cache_count += 1
                x = x_next
            else:
                x = self._analytic_update(x=x,t=t, dt=dt)

            if is_conditional:
                x = torch.where(condition, xT, x)  # keep conditioned positions unchanged
            
        return x

    def restore_model_and_cond_sample(self, num_steps, dataset, seq_len=100, prefix_len=50, eps=1e-5, duplicate=0, xT=None, given_t=None):        
        self._eval_mode()
        samples = self.generate_cond_samples(
            num_samples=self.config.loader.eval_batch_size,
            dataset=dataset, seq_len=seq_len, prefix_len=prefix_len,
            num_steps=num_steps,
            eps=eps,
            xT=xT,
            given_t=given_t)
        self._train_mode()
        return samples
    
    @torch.no_grad()
    def generate_cond_samples(self, num_samples, data, seq_len=100, prefix_len=50, 
                               num_steps=None, eps=1e-5, xT=None, given_t=None, seed=None):
        """Generate samples from the model."""
        xT = data.clone()
        unfix_len = (xT.shape[1] - prefix_len)
        xT[:, -unfix_len:] = -1
        fillin_mask = xT == -1
        xT = torch.where(fillin_mask, self.prior_sample(*xT.shape), xT)
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

    @torch.no_grad()
    def _semi_ar_sampler(
        self, n_samples, stride_length, num_strides, dt=0.001):
        # TODO(subham): Test this method after refactoring.
        ones = torch.ones(n_samples, dtype=self.dtype,
                                            device=self.device)

        num_steps = int(1 / dt)
        sampling_steps = 0
        intermediate_tokens = []
        target = None
        for _ in range(num_strides + 1):
            p_x0_cache = None
            x = self.prior_sample(n_samples, self.num_tokens)
            if target is not None:
                x[:, : -stride_length] = target
            for i in range(num_steps + 1):
                p_x0_cache, x_next = self._ancestral_update(
                    x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)
                if (not torch.allclose(x_next, x)
                        or self.time_conditioning):
                    p_x0_cache = None
                    sampling_steps += 1
                x = x_next
            x = self.forward(x, 0 * ones).argmax(dim=-1)
            intermediate_tokens.append(
                x[:, :stride_length].cpu().numpy())
            target = x[:, stride_length:]
        
        intermediate_tokens.append(target.cpu().numpy())
        intermediate_text_samples = []
        sequence_lengths = ((
            np.concatenate(intermediate_tokens, axis=1)[:, 1:]
            == self.tokenizer.eos_token_id).cumsum(-1) == 0).sum(-1)
        for i in range(2, len(intermediate_tokens) + 1):
            intermediate_text_samples.append(
                self.tokenizer.batch_decode(
                    np.concatenate(intermediate_tokens[:i], axis=1)))
        return (sampling_steps, intermediate_text_samples,
                        sequence_lengths)

    def restore_model_and_semi_ar_sample(
            self, stride_length, num_strides, dt=0.001):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        # TODO(subham): Test this method after refactoring.
        self._eval_mode()
        (sampling_steps, samples,
         sequence_lengths) = self._semi_ar_sampler(
            n_samples=self.config.loader.eval_batch_size,
            stride_length=stride_length,
            num_strides=num_strides, 
            dt=dt)
        self._train_mode()
        return sampling_steps, samples, sequence_lengths


class AbsorbingState(Diffusion):
    def __init__(self, config, tokenizer):
        # NOTE: Ideally, we should do 
        # vocab_size = len(tokenizer), so that we account
        # for the special tokens added in dataloader.py.
        # But we use tokenizer.vocab_size so as to to be
        # consistent with the prior checkpoints.
        vocab_size = tokenizer.vocab_size
        if (not hasattr(tokenizer, 'mask_token')
                or tokenizer.mask_token is None):
            self.mask_index = vocab_size
            vocab_size += 1
        else:
            self.mask_index = tokenizer.mask_token_id
        self.subs_masking = config.algo.subs_masking
        super().__init__(config, tokenizer,
                                         vocab_size=vocab_size)
        self.save_hyperparameters()

    def _validate_configuration(self):
        super()._validate_configuration()
        if self.parameterization in {'score', 'mean'}:
            assert self.time_conditioning
        assert not (self.parameterization == 'mean'
                                and self.T == 0)
        if self.T > 0:
            assert self.parameterization in {'mean', 'subs'}
        if self.subs_masking:
            assert self.parameterization == 'mean'

    def q_xt(self, x, alpha_t):
        """Computes the noisy sample xt.

        Args:
            x: int torch.Tensor with shape (batch_size,
                    diffusion_model_input_length), input. 
            alpha_t: float torch.Tensor with shape (batch_size, 1).
        """
        if self.config.algo.name == 'absorbbdd':
            assert 0
        move_indices = torch.rand(
            * x.shape, device=x.device) < 1 - alpha_t
        xt = torch.where(move_indices, self.mask_index, x)
        if self.ignore_bos:
            xt[:, 0] = x[:, 0]
        return xt

    def prior_sample(self, *batch_dims):
        return self.mask_index * torch.ones(
            * batch_dims, dtype=torch.int64, device=self.device)

    def _ancestral_update(self, x, t, dt, p_x0=None,
                                     noise_removal_step=False):
        _, alpha_t = self.noise(t)
        if noise_removal_step:
            alpha_s = torch.ones_like(alpha_t)
        else:
            _, alpha_s = self.noise(t - dt)
        assert alpha_t.ndim == 2
        if p_x0 is None:
            p_x0 = self.forward(
                x, self._sigma_from_alphat(alpha_t)).exp()
        
        q_xs = p_x0 * (alpha_s - alpha_t)[:, :, None]
        q_xs[:, :, self.mask_index] = 1 - alpha_s
        _x = sample_categorical(q_xs, self.config.sampling.temperature)
        
        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    def _staggered_score(self, score, dsigma):
        score = score.clone()
        extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., self.mask_index] += extra_const
        return score

    def _analytic_update(self, x, t, dt):
        sigma_t = self._sigma_from_alphat(self.noise(t)[1])
        sigma_s = self._sigma_from_alphat(self.noise(t - dt)[1])
        dsigma = sigma_t - sigma_s
        score = self._get_score(x, sigma_t)
        if self.config.sampling.use_float64:
            score = score.to(torch.float64)
        stag_score = self._staggered_score(score, dsigma)
        probs = stag_score * self._transp_transition(x, dsigma)
        return sample_categorical(probs)

    def _denoiser_update(self, x, t):
        sigma = self._sigma_from_alphat(self.noise(t)[1])
        score = self._get_score(x, sigma)
        if self.config.sampling.use_float64:
            score = score.to(torch.float64)
        stag_score = self._staggered_score(score, sigma)
        probs = stag_score * self._transp_transition(x, sigma)
        probs[..., self.mask_index] = 0
        samples = sample_categorical(probs)
        return samples

    def _transp_transition(self, i, sigma):
        sigma = _unsqueeze(sigma, reference=i[..., None])
        edge = torch.exp(-sigma) * F.one_hot(
            i, num_classes=self.vocab_size)
        edge += torch.where(i == self.mask_index,
                                                1 - torch.exp(-sigma).squeeze(-1),
                                                0)[..., None]
        return edge


class UniformState(Diffusion):
    def _validate_configuration(self):
        super()._validate_configuration()
        assert self.time_conditioning
        assert self.parameterization == 'mean'
        if self.config.algo.name != 'distillation':
            assert self.T == 0

    def q_xt(self, x, alpha_t):
        """Computes the noisy sample xt.

        Args:
            x: int torch.Tensor with shape (batch_size,
                    diffusion_model_input_length), input.
            move_chance: float torch.Tensor with shape
                (batch_size, 1).
        """
        move_indices = torch.rand(
            *x.shape, device=x.device) < 1 - alpha_t
        uniform_tensor = torch.randint(
            0, self.vocab_size, x.shape, device=x.device)
        xt = torch.where(move_indices, uniform_tensor, x)
        if self.ignore_bos:
            xt[:, 0] = x[:, 0]
        return xt

    def prior_sample(self, *batch_dims):
        return torch.randint(
            0, self.vocab_size, batch_dims, dtype=torch.int64,
            device=self.device)
