"""Console logger utilities.

Copied from https://github.com/HazyResearch/transformers/blob/master/src/utils/utils.py
Copied from https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
"""

import argparse
import logging
import os
import math
import sys
import pickle
import time
from math import isfinite
from typing import Union

import fsspec
import lightning
import numpy as np
from numpy.polynomial.hermite import hermgauss
import torch
from scipy.integrate import quad
from scipy.stats import norm
from scipy.special import log_ndtr  # stable log
from scipy.interpolate import CubicSpline  # Added for LUT/Spline
from timm.scheduler import CosineLRScheduler

def sigmoid(x):
    return 1./(1. + np.exp(-x))

def inv_sigmoid(y):
    return -np.log((1./y) - 1.)

def to_data_level(x):
    return sigmoid(-x) ** 0.5

def to_alpha(lut, value):
    data_level = to_data_level(value)
    alpha_val = lut(data_level)
    return alpha_val

def to_duo_gamma(lut, alpha):
    data_level = lut(alpha)
    gamma_val = -inv_sigmoid(data_level ** 2)
    return gamma_val

# ----------------------------
# Utilities: standardized means
# ----------------------------
def standardized_means(alpha: float, tau: float, b: float, diffusion=False):
    """
    Returns (m_c, m_u, m_a, sigma), where
      sigma = b * (1 - alpha),
      m_c = (alpha - tau) / sigma   (label / 'correct'),
      m_u = -tau / sigma            (other data),
      m_a = 0.0                     (absorbing)
    """
    if diffusion:
        sigma = b * (1.0 - alpha**2)
        sigma = sigma ** 0.5
    else:
        sigma = b * (1.0 - alpha)
    sigma = np.where(sigma <= 0.0, 1e-12, sigma)
    m_c = (alpha - tau) / sigma
    m_u = (-tau) / sigma
    m_a = 0.0 / sigma
    return m_c, m_u, m_a, sigma

def compute_qs_fast_np(gamma, tau, b: float, K: int, M: int, *,
                    n_gh: int = 100, sigma_floor: float = 1e-12,
                    diffusion=False):
    """
    Returns (q_c, q_u, q_a) vectorized for np.array inputs of gamma and tau.
    Uses log-stabilized Gauss–Hermite quadrature.

    Args:
        gamma: float or np.array shape (N,)
        tau: float or np.array shape (N,)
        b: float
        K: int
        M: int
        n_gh: number of quadrature nodes
        sigma_floor: minimum sigma value for numerical stability
        diffusion: boolean flag passed to standardized_means

    Returns:
        tuple of np.arrays (q_c, q_u, q_a), each with the same shape as gamma/tau.
    """
    # 1. Ensure inputs are numpy arrays (for consistency even if scalars are passed)
    gamma = np.asanyarray(gamma)
    tau = np.asanyarray(tau)

    # 2. Compute standardized means
    # NOTE: standardized_means must support numpy array broadcasting.
    # We assume it returns arrays of the same shape as gamma/tau.
    m_c, m_u, m_a, sigma = standardized_means(gamma, tau, b, diffusion)

    # 3. Vectorized sigma floor (replaces 'if sigma < sigma_floor')
    sigma = np.maximum(sigma, sigma_floor)

    # 4. GH nodes/weights
    # x, w shape: (n_gh,)
    x, w = hermgauss(n_gh)
    w = w / np.sqrt(np.pi)
    z_nodes = np.sqrt(2.0) * x  # Z ~ N(0,1)

    x = x[np.newaxis, ...]
    w = w[np.newaxis, ...]
    z_nodes = z_nodes[np.newaxis, ...]

    # 5. Expand dims for broadcasting
    # We want to broadcast (Batch, 1) against (n_gh,) -> resulting in (Batch, n_gh)
    # If inputs were scalars, these become (1, 1) or just (1) depending on dims,
    # but [..., np.newaxis] ensures the last dim aligns with z_nodes.
    m_c_exp = m_c[..., np.newaxis]
    m_u_exp = m_u[..., np.newaxis]
    # m_a is not used directly in logs below, but if needed: m_a_exp = m_a[..., np.newaxis]

    # --- Precompute LOG-CDFs ---
    # z_nodes shape: (1, n_gh)
    # means shape: (Batch, 1)
    # Resulting L shapes: (Batch, n_gh)

    # 0-shift
    L0 = log_ndtr(z_nodes)  # Shape (n_gh,) - broadcasts automatically later

    # label vs absorbing / absorbing vs label
    L_ca = log_ndtr(z_nodes + m_c_exp)
    L_ac = log_ndtr(z_nodes - m_c_exp)

    # data vs absorbing / absorbing vs data
    L_ua = log_ndtr(z_nodes + m_u_exp)
    L_au = log_ndtr(z_nodes - m_u_exp)

    # label vs data / data vs label
    d_cu = m_c_exp - m_u_exp
    L_cu = log_ndtr(z_nodes + d_cu)
    L_uc = log_ndtr(z_nodes - d_cu)

    # --- Build node-wise log-products ---
    # Shapes are preserved as (Batch, n_gh)

    # Label winner
    log_prod_c = (K - 1) * L_cu + M * L_ca

    # Wrong-data winner
    if K > 1:
        log_prod_u = L_uc + max(K - 2, 0) * L0 + M * L_ua
    else:
        log_prod_u = None

    # Absorbing winner
    if M > 0:
        log_prod_a = L_ac + (K - 1) * L_au + max(M - 1, 0) * L0
    else:
        log_prod_a = None

    # --- Weighted sum over nodes ---
    def weighted_exp_sum(logv):
        # Calculate sum over the last axis (the quadrature nodes)
        # w is (n_gh,), logv is (..., n_gh)
        # Broadcasting aligns w to the last dimension of logv
        return np.sum(w * np.exp(logv), axis=-1)

    q_c = weighted_exp_sum(log_prod_c)
    q_u = weighted_exp_sum(log_prod_u)

    # Handle conditional outputs ensuring correct array shapes
    if log_prod_a is not None and M > 0:
        q_a = (1 - q_c - q_u * (K-1))/M
    else:
        q_a = np.zeros_like(q_c)
        q_u = (1. - q_c)/(K - 1.)

    return q_c, q_u * (K-1), q_a * M

# ----------------------------
# Core: GH with precomputed log Φ-shifts (≤ 6 calls)
# ----------------------------
def compute_qs_fast(gamma: float, tau: float, b: float, K: int, M: int, *,
                    n_gh: int = 100, sigma_floor: float = 1e-12,
                    diffusion=False) -> tuple[float, float, float]:
    """
    Returns (q_c, q_u, q_a) using log-stabilized Gauss–Hermite and
    only a constant number of log_ndtr calls per evaluation.

    q_c : probability the label ('correct') class wins (per label)
    q_u : probability a particular non-label data class wins (per class)
    q_a : probability a particular absorbing class wins (per class)
    """
    # standardized means
    m_c, m_u, m_a, sigma = standardized_means(gamma, tau, b, diffusion)
    if sigma < sigma_floor:
        sigma = sigma_floor  # keep GH numerically sane; values remain consistent

    # GH nodes/weights for exp(-x^2); normalize to N(0,1)
    x, w = hermgauss(n_gh)
    w = w / np.sqrt(np.pi)
    z_nodes = np.sqrt(2.0) * x  # Z ~ N(0,1) evaluated at √2 x_ℓ

    # --- Precompute the LOG-CDFs for the few unique shifts we need ---
    # 0-shift (same-class competitors)
    L0     = log_ndtr(z_nodes)                # log Φ(z)
    # label vs absorbing / absorbing vs label
    L_ca   = log_ndtr(z_nodes + m_c)          # log Φ(z + (m_c - 0))
    L_ac   = log_ndtr(z_nodes - m_c)          # log Φ(z + (0   - m_c))
    # data vs absorbing / absorbing vs data
    L_ua   = log_ndtr(z_nodes + m_u)          # log Φ(z + (m_u - 0))
    L_au   = log_ndtr(z_nodes - m_u)          # log Φ(z + (0   - m_u))
    # label vs data / data vs label
    d_cu   = m_c - m_u
    L_cu   = log_ndtr(z_nodes + d_cu)         # log Φ(z + (m_c - m_u))
    L_uc   = log_ndtr(z_nodes - d_cu)         # log Φ(z + (m_u - m_c))

    # --- Build node-wise log-products for each grouped case ---
    # Label winner: (K-1) non-label data + M absorbing competitors
    #   log_prod_c(u) = (K-1)*log Φ(z + (m_c - m_u)) + M*log Φ(z + (m_c - 0))
    log_prod_c = (K - 1) * L_cu + M * L_ca

    # Wrong-data winner (per class): 1 label + (K-2) other data + M absorbing
    #   log_prod_u(u) = log Φ(z + (m_u - m_c)) + (K-2)*log Φ(z) + M*log Φ(z + (m_u - 0))
    if K > 1:
        log_prod_u = L_uc + max(K - 2, 0) * L0 + M * L_ua
    else:
        log_prod_u = None  # no wrong-data class exists

    # Absorbing winner (per class): 1 label + (K-1) data + (M-1) absorbing
    #   log_prod_a(u) = log Φ(z + (0 - m_c)) + (K-1)*log Φ(z + (0 - m_u)) + (M-1)*log Φ(z)
    if M > 0:
        log_prod_a = L_ac + (K - 1) * L_au + max(M - 1, 0) * L0
    else:
        log_prod_a = None  # no absorbing class exists

    # --- Weighted sum over nodes; clip exponents for safety ---
    def weighted_exp_sum(logv):
        # return float(np.sum(w * np.exp(np.clip(logv, -745.0, 745.0))))  # -745 ~ float64 underflow
        return float(np.sum(w * np.exp(logv)))  # -745 ~ float64 underflow

    q_c = weighted_exp_sum(log_prod_c)
    q_u = weighted_exp_sum(log_prod_u) if (log_prod_u is not None and K > 1) else 0.0
    q_a = weighted_exp_sum(log_prod_a) if (log_prod_a is not None and M > 0) else 0.0

    return q_c, q_u, q_a


# ----------------------------
# Core Exact Computation (Gamma -> Alpha)
# ----------------------------
def compute_alpha_exact(gamma: np.ndarray, K: int, n_gh: int = 100, sigma_floor: float = 1e-12, is_diffusion=False) -> np.ndarray:
    """
    Computes q_c (Alpha) from Gamma using Gauss-Hermite integration.
    This is the ground-truth function mapping Gamma -> Alpha.
    """
    gamma = np.asarray(gamma)

    # 1. Standardized means (assuming tau=0, b=1.0 for this conversion)
    if is_diffusion:
        sigma = 1.0 - gamma**2    
        sigma = np.sqrt(sigma)
    else:
        sigma = 1.0 - gamma
    sigma = np.maximum(sigma, sigma_floor)
    
    m_c = gamma / sigma
    
    # 2. GH nodes/weights
    x, w = hermgauss(n_gh)
    w = w / np.sqrt(np.pi)
    z_nodes = np.sqrt(2.0) * x

    # 3. Broadcasting
    m_c_expanded = m_c[:, None]   # (B, 1)
    z_expanded = z_nodes[None, :] # (1, n_gh)

    # 4. Compute Log-CDFs
    # L_cu = log(Phi(z + m_c))
    L_cu = log_ndtr(z_expanded + m_c_expanded)

    # 5. Weighted sum
    # log_prod_c = (K - 1) * L_cu
    log_prod_c = (K - 1) * L_cu
    q_c = np.sum(w * np.exp(log_prod_c), axis=-1)
    
    # Debugged. should consider prob. from uniform noise.
    alpha = K/(K-1.) * (q_c - 1./K)

    alpha += (gamma-1) * 1e-10 # minor trick to ensure monotonicity

    alpha = np.clip(alpha, 0.0, 1.0)

    return alpha

# ----------------------------
# LUT / Spline Implementation
# ----------------------------

def build_luts(K: int, n_points: int = 10000, is_diffusion=False) -> tuple[CubicSpline, CubicSpline]:
    """
    Builds two lookup tables (Splines):
    1. Alpha -> Gamma (Forward)
    2. Gamma -> Alpha (Inverse)
    
    Reverted to Linear (Uniform) spacing.
    Chebyshev nodes concentrate points at 0 and 1, but for large K, the curve 
    is often sigmoid-like (flat at ends, steep in middle). 
    Uniform spacing captures the transition region better.
    """
    # 1. Create Alpha grid using Uniform Spacing
    # Simple linspace covers the whole range evenly.
    gamma_vals = np.linspace(0.0, 1.0, n_points) # cont.
    
    # 2. Compute corresponding Gamma grid (Exact)
    alpha_vals = compute_alpha_exact(gamma_vals, K=K, is_diffusion=is_diffusion) # disc.
    
    # 3. Build Forward Spline (Alpha -> Gamma)
    # Alpha is strictly increasing. Safe.
    lut_g2a = CubicSpline(gamma_vals, alpha_vals)
    
    # 4. Build Inverse Spline (Gamma -> Alpha)
    # Gamma values must be strictly increasing to be 'x' in CubicSpline.
    
    # Sort just in case (though usually monotonic)
    sorted_indices = np.argsort(alpha_vals)
    gamma_sorted = gamma_vals[sorted_indices]
    alpha_sorted = alpha_vals[sorted_indices]
    
    # Remove duplicates in Gamma
    # Duplicates often happen at very low alpha (gamma ~ 1/K) or very high alpha (gamma ~ 1.0)
    unique_alpha, unique_indices = np.unique(alpha_sorted, return_index=True)
    unique_gamma = gamma_sorted[unique_indices]

    # Create Spline
    lut_a2g = CubicSpline(unique_alpha, unique_gamma)
    
    return lut_a2g, lut_g2a

def build_luts_ext(K: int, M: int, margin_func, n_points: int = 10000, is_diffusion=False):
    """
    Builds two lookup tables (Splines):
    1. Alpha -> Gamma (Forward)
    2. Gamma -> Alpha (Inverse)
    3. Alpha -> Beta

    Reverted to Linear (Uniform) spacing.
    Chebyshev nodes concentrate points at 0 and 1, but for large K, the curve
    is often sigmoid-like (flat at ends, steep in middle).
    Uniform spacing captures the transition region better.
    """
    # 1. Create Gamma grid using Uniform Spacing
    gamma_vals = np.linspace(0.0, 1.0, n_points) # cont.

    # 2. Compute corresponding Alpha and Beta grid
    tau_vals = margin_func(gamma_vals)
    q_c_computed_vals, qu_raw_vals, qa_raw_vals = compute_qs_fast_np(
        gamma_vals,
        tau=tau_vals,
        b=1.0,
        K=K,
        M=M,
        diffusion=is_diffusion
    )

    lambda_t = qu_raw_vals / (1 - q_c_computed_vals+1e-12)
    
    print('Max lambda_t:', np.max(lambda_t))

    # Clip q_c_computed_vals to [0, 1] to prevent numerical issues, as they represent probabilities
    q_c_computed_vals = np.clip(q_c_computed_vals, 0.0, 1.0)

    # `beta_vals` here corresponds to q_u * (K-1)
    qu_raw_vals = np.clip(qu_raw_vals, 0.0, 1.0)
    beta_vals = qu_raw_vals * (K/(K - 1.0))
    beta_vals = np.clip(beta_vals, 0.0, 1.0)
    
    alpha_vals = q_c_computed_vals - 1./K * beta_vals
    alpha_vals = np.clip(alpha_vals, 0.0, 1.0)

    # Sort alpha_vals and gamma_vals for inverse mapping
    # Here, alpha_vals is the 'alpha' in LUT_A2G/LUT_G2A naming context
    # gamma_vals is the 'gamma' in LUT_A2G/LUT_G2A naming context
    sorted_indices = np.argsort(alpha_vals)
    alpha_sorted = alpha_vals[sorted_indices]
    gamma_sorted = gamma_vals[sorted_indices]
    beta_sorted = beta_vals[sorted_indices] # Also sort beta_vals

    # Filter for unique alpha values to ensure strictly increasing x for CubicSpline
    # and to avoid extremely small differences that cause numerical instability
    unique_alpha, unique_indices = np.unique(alpha_sorted, return_index=True)
    unique_gamma = gamma_sorted[unique_indices]
    unique_beta = beta_sorted[unique_indices]

    # Additional filtering to ensure sufficient difference between unique_alpha points
    # This addresses the 'dydx must contain only finite values' error by preventing dx=0
    min_diff = 1e-8 # A small threshold for differences in x values

    filtered_alpha = [unique_alpha[0]]
    filtered_gamma = [unique_gamma[0]]
    filtered_beta = [unique_beta[0]]

    for i in range(1, len(unique_alpha)):
        if unique_alpha[i] - filtered_alpha[-1] > min_diff:
            filtered_alpha.append(unique_alpha[i])
            filtered_gamma.append(unique_gamma[i])
            filtered_beta.append(unique_beta[i])
    
    # Ensure filtered lists are numpy arrays
    filtered_alpha = np.array(filtered_alpha)
    filtered_gamma = np.array(filtered_gamma)
    filtered_beta = np.array(filtered_beta)

    # Build LUT G2A: gamma_input_vals -> q_c_computed_vals (Gamma -> Alpha)
    # gamma_vals are strictly increasing by definition of linspace, so this spline should be stable.
    lut_g2a = CubicSpline(gamma_vals, q_c_computed_vals)

    # Build LUT A2G: q_c_computed_vals -> gamma_input_vals (Alpha -> Gamma)
    # And LUT A2B: q_c_computed_vals -> beta_vals (Alpha -> Beta)
    # Use the filtered points for these splines to ensure stability
    lut_a2g = CubicSpline(filtered_alpha, filtered_gamma)
    lut_a2b = CubicSpline(filtered_alpha, filtered_beta)

    return lut_a2g, lut_g2a, lut_a2b

# Initialize LUTs globally (lazy loading or explicit init recommended in real apps, 
# but running here for immediate use)
# Using a default K=50000 as per previous context.

# LUT_A2G, LUT_G2A = build_luts(K=50000)

# Disc. to Cont.

def pull_lut(x, lut):
    if isinstance(x, torch.Tensor):
        y = np.clip(lut(x.cpu().numpy()), 0.0, 1.0)
        return torch.from_numpy(y).to(x.device)
    else:
        return np.clip(lut(x), 0.0, 1.0)

def alpha_to_gamma(alpha: Union[np.ndarray, torch.tensor], lut: CubicSpline) -> Union[np.ndarray, torch.tensor]:
    """
    Maps Alpha -> Gamma using the LUT.
    """
    if isinstance(alpha, torch.Tensor):
        gamma = np.clip(lut(alpha.cpu().numpy()), 0.0, 1.0)
        return torch.from_numpy(gamma).to(alpha.device)
    else:
        return np.clip(lut(alpha), 0.0, 1.0)

# Cont. to Disc.
def gamma_to_alpha(gamma: Union[np.ndarray, torch.tensor], lut: CubicSpline) -> Union[np.ndarray, torch.tensor]:
    """
    Maps Gamma -> Alpha using the LUT.
    """
    # Clip result to [0, 1] to avoid spline overshoot
    if isinstance(gamma, torch.Tensor):
        alpha = np.clip(lut(gamma.cpu().numpy()), 0.0, 1.0)
        return torch.from_numpy(alpha).to(gamma.device)
    else:
        return np.clip(lut(gamma), 0.0, 1.0)

def count_parameters(model):
    return sum(p.numel()
                         for p in model.parameters()
                         if p.requires_grad)

def fsspec_exists(filename):
    """Check if a file exists using fsspec."""
    fs, _ = fsspec.core.url_to_fs(filename)
    return fs.exists(filename)


def fsspec_listdir(dirname):
    """Listdir in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(dirname)
    return fs.ls(dirname)


def fsspec_mkdirs(dirname, exist_ok=True):
    """Mkdirs in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(dirname)
    fs.makedirs(dirname, exist_ok=exist_ok)


def print_nans(tensor, name):
    if torch.isnan(tensor).any():
        print(name, tensor)


class LRHalveScheduler:
    def __init__(self, warmup_steps, n_halve_steps):
        self.warmup_steps = warmup_steps
        self.n_halve_steps = n_halve_steps
    
    def __call__(self, current_step):
        if current_step < self.warmup_steps:
            return current_step / self.warmup_steps
        return 0.5 ** ((current_step - self.warmup_steps)
                                     // self.n_halve_steps)


class CosineDecayWarmupLRScheduler(
    CosineLRScheduler,
    torch.optim.lr_scheduler._LRScheduler):
    """Wrap timm.scheduler.CosineLRScheduler
    Enables calling scheduler.step() without passing in epoch.
    Supports resuming as well.
    Adapted from:
        https://github.com/HazyResearch/hyena-dna/blob/main/src/utils/optim/schedulers.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_epoch = -1
        self.step(epoch=0)

    def step(self, epoch=None):
        if epoch is None:
            self._last_epoch += 1
        else:
            self._last_epoch = epoch
        # We call either step or step_update, depending on
        # whether we're using the scheduler every epoch or every
        # step.
        # Otherwise, lightning will always call step (i.e.,
        # meant for each epoch), and if we set scheduler
        # interval to "step", then the learning rate update will
        # be wrong.
        if self.t_in_epochs:
            super().step(epoch=self._last_epoch)
        else:
            super().step_update(num_updates=self._last_epoch)


class LoggingContext:
    """Context manager for selective logging."""
    def __init__(self, logger, level=None, handler=None, close=True):
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()


class GradientInspectionCallback(lightning.Callback):
        def __init__(self, num_grads_log):
                self.num_grads_log = 10

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            gradients = []
            for name, param in pl_module.backbone.blocks.named_parameters():
                    gradients.append(param.grad.view(-1))

            if gradients:
                grads = torch.cat((gradients))
                if not hasattr(pl_module, 'grad_accum_buffer'):
                    pl_module.grad_step = torch.tensor(
                        0, device=pl_module.device)
                    pl_module.grad_accum_buffer = torch.zeros(
                        self.num_grads_log,
                        grads.shape[0],
                        device=pl_module.device)
                pl_module.grad_accum_buffer[pl_module.grad_step] = grads
                pl_module.grad_step += 1

            if (hasattr(pl_module, 'grad_accum_buffer') 
                    and pl_module.grad_step == self.num_grads_log):
                grads = pl_module.grad_accum_buffer
                grad_var = grads.std(0).mean()
                pl_module.log(name='trainer/grad_var',
                                            value=grad_var.item(),
                                            on_step=True,
                                            on_epoch=False,
                                            sync_dist=True)
                # import ipdb; ipdb.set_trace()
                # should save the grads tensor as a numpy array
                # and visualize mean, median, top-k
                pl_module.grad_accum_buffer.zero_()
                pl_module.grad_step = 0


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ('debug', 'info', 'warning', 'error',
                                'exception', 'fatal', 'critical'):
        setattr(logger,
                        level,
                        lightning.pytorch.utilities.rank_zero_only(
                            getattr(logger, level)))

    return logger


# Copied from https://github.com/jdeschena/sdtt/blob/bbc54d5b3c5fcffd79602cff17ed34dde1f3eff6/src/sdtt/core/sampling/utils.py#L10
def top_k_top_p_filtering(
        logits,
        top_k=0,
        top_p=0.0,
        filter_value=-float("Inf"),
        dim=-1):
        """Filter a distribution of logits using top-k/top-p (nucleus) filtering.
        Adapted from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

        Args:
            logits (Tensor): Tensor of logits
            top_k (int, optional): Number of top values to keep.
                    Deactivated if k is 0. Defaults to 0.
            top_p (float, optional): Cumulative mass to retain.
                    Deactivated if p = 0. Defaults to 0.0.
            filter_value (float, optional): Fill value to replace
                    the entries removed by top-k/top-p filtering.
                    Defaults to -float('Inf').
            dim (int, optional): Dimension of the filtering. Defaults to -1.

        Returns:
                logits: Tensor whose axis `dim` was filtered.
        """
        if dim != -1:
            logits = torch.transpose(logits, dim, -1)

        assert top_k < logits.size(dim)
        if top_k > 0:
            # Remove all tokens with a probability less than
            # the last token of the top-k
            values, _ = torch.topk(logits, k=top_k, dim=-1)
            to_remove_mask = (
                    logits < torch.min(values, dim=-1, keepdim=True)[0]
            )  # min returns a tuple (values, indices)
            logits[to_remove_mask] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(
                logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cum_probs > top_p
            # Ensures at least one token is kept
            sorted_indices_to_remove[..., 1:] = \
                sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            mask_to_remove = torch.empty_like(sorted_indices_to_remove)
            mask_to_remove.scatter_(dim=-1,
                                                            index=sorted_indices,
                                                            src=sorted_indices_to_remove)
            logits[mask_to_remove] = filter_value

        if dim != -1:
            logits = torch.transpose(logits, dim, -1)

        return logits


def _discrete_prob_map(gamma_t, N=10):
    snr_sqrt = np.exp(-gamma_t / 2)
    def value(x):
        cdf = norm.cdf(x, scale=1) ** (N - 1)
        pdf = norm.pdf(x, loc=snr_sqrt, scale=1)
        return pdf * cdf
    return value


def _discrete_prob_grad(gamma_t, N=10):
    snr_sqrt = np.exp(-gamma_t / 2)
    def value(x):
        coef = -0.5 * snr_sqrt * (x - snr_sqrt)
        cdf = norm.cdf(x, scale=1) ** (N - 1)
        pdf = norm.pdf(x, loc=snr_sqrt, scale=1)
        return coef * pdf * cdf
    return value


def _cache_prob_usdm_in_partition(
    vocab_size=30522, partition_index=0, num_partitions=1,
    log10_num_points=5):
    print(f'Caching partition:{partition_index} / {num_partitions}')
    path = 'integral'
    gamma_min = -5
    gamma_max = -1
    num_points = 10 ** log10_num_points
    p_cache = []
    grad_p_cache = []
    start_time = time.time()
    gammas = np.linspace(gamma_min, gamma_max, num_points)
    n = num_points // num_partitions
    for gamma in gammas[partition_index * n:
                                            (partition_index + 1) * n]:
        pt, _ = quad(_discrete_prob_map(gamma, vocab_size),
                                 -np.inf, np.inf)
        p_cache.append(pt)
        grad_pt, _ = quad(_discrete_prob_grad(gamma, vocab_size),
                                            -np.inf, np.inf)
        grad_p_cache.append(grad_pt)
        if len(p_cache) % 100 == 0:
            print('{}% completed. Time elapsed:{:.2f} mins'.format(
                int(100 * len(p_cache) / num_points),
                (time.time() - start_time) / 60))

    filename = os.path.join(
        path, '{}_{}_{}-{}.pkl'.format(
            vocab_size, log10_num_points, partition_index,
            num_partitions))
    with open(filename, 'wb') as f:
        pickle.dump({
            'vocab_size': vocab_size,
            'gamma_min': gamma_min,
            'gamma_max': gamma_max,
            'num_points': num_points,
            'pt': np.asarray(p_cache),
            'grad_pt': np.asarray(grad_p_cache)}, f)


def test_cache_prob_usdm_in_partition(
    partition_index=0, num_partitions=1, vocab_size=30522,
    log10_num_points=5):
    path = 'integral/{}_{}_{}-{}.pkl'.format(
        vocab_size, log10_num_points, partition_index,
        num_partitions)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    num_points = data['num_points']
    def _get_index(x):
        return round((num_points - 1) * (x - data['gamma_min']) / (
            data['gamma_max'] - data['gamma_min']))

    pt_errors = []
    grad_pt_errors = []
    gammas = np.linspace(data['gamma_min'],
                                             data['gamma_max'],
                                             num_points)
    n = num_points // num_partitions
    for gamma in gammas[partition_index * n:
                                            (partition_index + 1) * n]:
        pt, _ = quad(
            _discrete_prob_map(gamma, data['vocab_size']),
            -np.inf, np.inf)
        grad_pt, _ = quad(
            _discrete_prob_grad(gamma, data['vocab_size']),
            -np.inf, np.inf)
        idx = _get_index(gamma)
        print(idx)
        pt_errors.append((pt - data['pt'][idx]) ** 2)
        grad_pt_errors.append((grad_pt - data['grad_pt'][idx]) ** 2)
    print('Integral MSE:{} Integral Squared:{:.4f}'.format(
        np.mean(pt_errors), np.mean(data['pt'] ** 2)))
    print('Integral Grad MSE:{} Integral Grad Squared:{:.4f}'.format(
        np.mean(grad_pt_errors), np.mean(data['grad_pt'] ** 2)))

if __name__ == "__main__":
    # Usage: python utils.py --vocab_size=N
    parser = argparse.ArgumentParser(
        description='Caches the integral appearing in the '
                                'Diffusion Transformation operator.')
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=50257,  # For the gpt2 tokenizer
        help='Vocabulary size (default: 50257)')
    parser.add_argument(
        '--partition_index',
        type=int,
        default=0,
        help='Helps parallelize caching')
    parser.add_argument(
        '--num_partitions',
        type=int,
        default=1,
        help='Helps parallelize caching')
    parser.add_argument(
        '--log10_num_points',
        type=int,
        default=5,
        help=('The integral is function that needs to be '
                    'evaluated for inputs with a range [-5, 1]. '
                    'This argument represents the logarithm base 10 '
                    'of number of bins of discretization.'))
    args = parser.parse_args()

    # Computing the integral over [-5, 1] can be slow,
    # so one might prefer splitting it into `num_partitions`
    # bins and compute each separately and merge them later.
    _cache_prob_usdm_in_partition(
        partition_index=args.partition_index,
        num_partitions=args.num_partitions,
        vocab_size=args.vocab_size,
        log10_num_points=args.log10_num_points)
    
    test_cache_prob_usdm_in_partition(
        partition_index=args.partition_index,
        num_partitions=args.num_partitions,
        vocab_size=args.vocab_size,
        log10_num_points=args.log10_num_points)
    
    
    
# Util functions for topk approximation

def sample_gaussian_topk(B, N, k, sigma=1.0, device='cuda'):
    """
    Algorithm 2: Reverse Sampling from Order Statistics of Gaussian Random Variables
    
    Args:
        B (int): 배치 크기 (Batch size)
        N (int): 전체 변수의 개수 (Total number of variables)
        k (int): 뽑고 싶은 상위 개수 (Number of top values)
        sigma (float) or (B, 1): 정규분포 표준편차
        device (str): 실행할 디바이스
        
    Returns:
        torch.Tensor: 상위 k개의 가우시안 노이즈 (내림차순 정렬됨: 1등 -> k등)
    """
    # 수치적 정밀도를 위해 float64 사용 권장 (N이 클 때 log sum 연산 오차 방지)
    dtype_calc = torch.float64
    
    # 1. Sample U_l ~ Uniform(0, 1) for l from N down to N-k+1
    U = torch.rand((B, k), device=device, dtype=dtype_calc)
    
    # Indices l: [N, N-1, ..., N-k+1]
    # 알고리즘 수식의 l에 해당
    l_indices = torch.arange(N, N - k, -1, device=device, dtype=dtype_calc)
    
    # 2. Compute random variables R_l = log(U_l) / l
    # (Renyi Representation)
    R = torch.log(U) / l_indices.unsqueeze(0)  # shape (B, k)
    
    # 3. Compute cumulative sums P_l
    P = torch.cumsum(R, dim=1)
    
    delta = -torch.expm1(P) # 1-V
    delta = delta.clamp(min=1e-30, max=1.0-1e-30)
    
    # 4. Let V_l = exp(P_l)
    # V = torch.exp(P) # shape (B, k)
    
    # 5. Apply inverse normal CDF: X = Phi^-1(V) * sigma
    # torch.special.ndtri is Phi^-1
    # V is very close to 1, so keep float64 to avoid inf in float32, then convert
    # X = torch.special.ndtri(V) * sigma
    X = -torch.special.ndtri(delta) * sigma # shape (B, k)
    return X # shape (B, k)


def sample_topk_gaussians_exact(k, total_vocab, sigma):
    noise = torch.randn(total_vocab) * sigma
    topk_noise = torch.topk(noise, k).values
    return topk_noise


def calculate_conditional_logmean(c, sigma, tau):
    """
    μ = E[exp(z/τ) | z < c] for z ~ N(0, σ)
    """
    # Formula components:
    # sigma = sigma / tau
    # term1 = sigma ** 2 / 2
    # term2 = - log(Phi(c / sigma))
    # term3 = + log(Phi((c - sigma^2) / sigma))
    
    assert c.ndim == sigma.ndim, "c and sigma must have the same number of dimensions"

    sigma_scaled = sigma / tau
    term1 = sigma_scaled ** 2 / 2.0
    term2 = - torch.special.log_ndtr(c / sigma_scaled)
    term3 = torch.special.log_ndtr((c - sigma_scaled ** 2) / sigma_scaled)
    
    log_mu = term1 + term2 + term3        
    
    return log_mu

def sample_neq_x(x, k, vocab_size):
    """
    sample indices not equal to x
    """
    B = x.shape[0]
    device = x.device
    noise_indices = torch.randint(0, vocab_size - 1, (B, k), device=x.device)
    
    # if noise_indices >= x, +1 (exclude x)
    mask = noise_indices >= x.unsqueeze(-1)
    noise_indices = noise_indices + mask.int()
    
    # Rejection sampling for without replacement (maximum 10 tries)
    for _ in range(10):
        sorted_noise, _ = torch.sort(noise_indices, dim=1)
        # Check for duplicates in each row
        has_duplicate = (sorted_noise[:, 1:] == sorted_noise[:, :-1]).any(dim=1)
        if not has_duplicate.any():
            break
        # Resample duplicates
        mask_retry = has_duplicate
        x_retry = x[mask_retry]
        num_retry = mask_retry.sum().item()
        
        # resampling
        new_sample = torch.randint(0, vocab_size - 1, (num_retry, k), device=device)
        mask_shift_retry = new_sample >= x_retry.unsqueeze(1)
        new_sample = new_sample + mask_shift_retry.int()
        
        noise_indices[mask_retry] = new_sample
    return noise_indices

def scalable_topk_approximation(x, vocab_size, k, tau, alpha_t, sigma_t):
    """
    Algorithm 1 Implementation
    
    Args:
        x (Tensor): Correct token indices (B,)
        vocab_size (int): K
        k (int): Top-k parameter
        tau (float): Temperature
        alpha_t (float/Tensor (B,1)): Mean for clean data
        sigma_t (float/Tensor (B,1)): Std dev schedule
    
    Returns:
        log_softmax_logit (Tensor): LogSoftmax weights (Batch, k)
        tilde_x (Tensor): Top-k indices (Batch, k)
        zt (Tensor): Index of the largest variable (Batch,)
    """
    assert x.ndim == 1, "x should be of shape (B,)"
    B = x.shape[0]
    device = x.device
    
    # 1. Sample Top-k Gaussians (Noise) -> {z_0^(i)}
    # vocab_size -1 to exclude clean data
    z_noise_topk = sample_gaussian_topk(B, vocab_size - 1, k, sigma_t, device) # (B, k)
    # if torch.isnan(z_noise_topk).any():
    #     import ipdb; ipdb.set_trace(context=16)
    #     raise ValueError("NaN values found in sampled top-k Gaussian noise.")
    
    # 2. Sample Clean Data Index -> z_alpha
    # z_alpha ~ N(alpha_t, sigma_t)
    z_alpha = torch.normal(mean=alpha_t, std=sigma_t) # (B, 1)
    # if torch.isnan(z_alpha).any():
    #     import ipdb; ipdb.set_trace(context=16)
    #     raise ValueError("NaN values found in sampled clean data index.")
    
    # 3. Z_top <- top-k({z_alpha} U {z_0})
    # Concatenate noise and signal
    candidates_val = torch.cat([z_noise_topk, z_alpha], dim=-1) # (B, k+1)
    # if torch.isnan(candidates_val).any():
    #     import ipdb; ipdb.set_trace(context=16)
    #     raise ValueError("NaN values found in candidate values.")
    
    # z_alpha at index k
    # Top-k selection
    z_top_val, z_top_idx = torch.topk(candidates_val, k=k, dim=-1) # (B, k)
    # if torch.isnan(z_top_val).any():
    #     import ipdb; ipdb.set_trace(context=16)
    #     raise ValueError("NaN values found in top-k selected values.")
    
    # 4. mu calculation for normalization
    # c = min(Z_top) for each batch
    c = z_top_val.min(dim=-1, keepdim=True).values # (B, 1) to match dim with sigma_t
    logmu = calculate_conditional_logmean(c, sigma_t, tau) # (B,)
    # if torch.isnan(logmu).any():
    #     import ipdb; ipdb.set_trace(context=16)
    #     raise ValueError("NaN values found in logmu calculation.")
    
    # 5. Calculate partial Sum S
    # S = sum(exp(Z_top / tau))
    logS = torch.logsumexp(z_top_val / tau, dim=-1, keepdim=True) # (B,)
    # if torch.isnan(logS).any():
    #     import ipdb; ipdb.set_trace(context=16)
    #     raise ValueError("NaN values found in logS calculation.")
    
    # 6. If/Else Logic (Vectorized)
    # Condition: z_alpha (index k in candidates) is in z_top_idx?
    is_alpha_in_top = (z_top_idx == k).any(dim=-1).unsqueeze(-1) # (B,) boolean mask
    # if torch.isnan(is_alpha_in_top.float()).any():
    #     import ipdb; ipdb.set_trace(context=16)
    #     raise ValueError("NaN values found in is_alpha_in_top calculation.")
    
    # K: vocab_size
    # Tail approximation term
    const_in = (vocab_size - k)
    const_out = (vocab_size - k - 1)
    
    # Update S
    # Case 1 (In top): S += (K-k)*mu
    # Case 2 (Out top): S += (K-k-1)*mu + exp(z_alpha/tau)
    
    term_in = math.log(const_in) + logmu  # log((K-k)*mu)
    term_out = math.log(const_out) + logmu  # log((K-k-1)*mu)
    
    total_term_in = torch.logaddexp(logS, term_in)
    total_term_out = torch.logaddexp(logS, torch.logaddexp(term_out, z_alpha / tau))
    
    # if torch.isnan(total_term_in).any():
    #     import ipdb; ipdb.set_trace(context=16)
    #     raise ValueError("NaN values found in total_term_in calculation.")
    # if torch.isnan(total_term_out).any():
    #     import ipdb; ipdb.set_trace(context=16)
    #     raise ValueError("NaN values found in total_term_out calculation.")
    
    logS = torch.where(is_alpha_in_top, total_term_in, total_term_out) # (B, 1)
    
    log_softmax_logit = z_top_val / tau - logS  # (B, k), log(exp(z_i / tau) / S)
    # if torch.isnan(log_softmax_logit).any():
    #     import ipdb; ipdb.set_trace(context=16)
    #     raise ValueError("NaN values found in log_softmax_logit calculation.")

    # 7. Construct tilde_x (Indices)
    # Sample k noise indices (not equal to x)
    replace_indices = torch.full((B, k), False, device=device)
    replace_indices[z_top_idx == k] = True  # If alpha in top, replace index of z_alpha with x
    noise_indices = sample_neq_x(x, k, vocab_size) # (B, k)
    tilde_x = torch.where(replace_indices, x.unsqueeze(-1), noise_indices) # (B, k)
    
    # 8. Construct zt (Index of the largest variable)
    local_argmax = z_top_val.argmax(dim=-1) # (B,)
    zt = tilde_x[torch.arange(B, device=device), local_argmax] # (B,)
    
    unit_val = logmu-logS
    
    return log_softmax_logit, tilde_x, zt, unit_val  # return per-vocab log_softmax_logit.


def robust_seeded_randn(*size, seed=None, device='cpu', mode='rand'):
    """
    batch_seeds: (B,)
    size: shape except batch dimension
    """
    shape = size
    B = seed.shape[0]
    assert seed.ndim == 1 and seed.dtype == torch.int64, "seed should be of shape (B,) and dtype int64"
    
    # seed to long int (B, 1, 1, ...)
    view_dims = [B] + [1] * (len(shape))
    seeds_int = seed.view(*view_dims).to(device).long()
    
    # 2. create spatial idx (int64)
    num_spatial_elements = 1
    for dim in shape:
        num_spatial_elements *= dim
        
    spatial_idx = torch.arange(num_spatial_elements, device=device)
    spatial_view_dims = [1] + list(shape)
    spatial_idx = spatial_idx.view(*spatial_view_dims).long() # (1, D1, D2, ...)
    
    # 3. Bitwise Mixing (PCG Hash)
    # shuffle bits to get uniform distribution
    
    # Magic Constants
    M1 = 0xd251125f
    M2 = 0xcd9e8d57
    
    # 3-1. Uniform Noise (U1)
    state1 = (seeds_int * M1) ^ (spatial_idx * M2) 
    state1 = (state1 ^ (state1 >> 15)) * M1
    state1 = (state1 ^ (state1 >> 13)) * M2
    state1 = state1 ^ (state1 >> 16)
    
    # 3-2. Uniform Noise (U2)
    # offset=12345 to decorrelate from U1
    state2 = (seeds_int * M2) ^ ((spatial_idx + 12345) * M1) 
    state2 = (state2 ^ (state2 >> 15)) * M2
    state2 = (state2 ^ (state2 >> 13)) * M1
    state2 = state2 ^ (state2 >> 16)
    
    # 4. Int64 -> Float (0.0 ~ 1.0)
    # map to (0, 1) with bit masking and division
    # 0xFFFFFFFF = 2^32 - 1
    u1 = (state1 & 0xFFFFFFFF).float() / 4294967296.0
    u2 = (state2 & 0xFFFFFFFF).float() / 4294967296.0
    
    if mode == 'rand':
        return u1
    
    # clamp for stability (prevent log(0))
    u1 = torch.clamp(u1, min=1e-8, max=1.0)
    u2 = torch.clamp(u2, min=1e-8, max=1.0)
    
    # 5. Box-Muller Transform
    if mode == 'randn':
        radius = torch.sqrt(-2.0 * torch.log(u1))
        theta = 2.0 * torch.pi * u2
        
        return radius * torch.cos(theta)
    else:
        raise ValueError(f'Unknown mode: {mode}')


def convert_ema_state_dict(ckpt):
    orig_state_dict = ckpt['state_dict']
    ema_params = ckpt['ema']['shadow_params']
    
    new_state_dict = {}
    i = 0
    for key in orig_state_dict.keys():
        if key.endswith('backbone.rotary_emb.inv_freq'):
            print(f"Skipping Non-trainable parameter: {key}")
            new_state_dict[key] = orig_state_dict[key]
            continue
        else:
            # shape check
            if orig_state_dict[key].shape != ema_params[i].shape:
                # For SDTT reproduce code
                if orig_state_dict[key].requires_grad == False:
                    print(f"Warning: Shape mismatch for non-trainable parameter {key}, "
                          f"keeping original value. "
                          f"orig {orig_state_dict[key].shape} vs "
                          f"ema {ema_params[i].shape}")
                    new_state_dict[key] = orig_state_dict[key]
                    continue
                raise ValueError(f"Shape mismatch for key {key}: "
                                         f"orig {orig_state_dict[key].shape} vs "
                                         f"ema {ema_params[i].shape}")
            new_state_dict[key] = ema_params[i]
            i += 1
    assert len(new_state_dict) == len(orig_state_dict), "State dict size mismatch after conversion"
    return new_state_dict
        
