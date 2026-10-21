"""Linear diffusion schedule and the forward noising process."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class NoiseSchedule:
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_bar: jnp.ndarray


def make_schedule(
    num_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    dtype=jnp.float32,
) -> NoiseSchedule:
    # The product has a thousand factors, so form it in float64 and then store
    # the training dtype. A float32 running product drifts off the sheet's value.
    betas = np.linspace(beta_start, beta_end, num_steps, dtype=np.float64)
    alphas = 1.0 - betas
    alpha_bar = np.concatenate([np.ones(1, dtype=np.float64), np.cumprod(alphas)])
    return NoiseSchedule(
        betas=jnp.asarray(betas, dtype=dtype),
        alphas=jnp.asarray(alphas, dtype=dtype),
        alpha_bar=jnp.asarray(alpha_bar, dtype=dtype),
    )


def add_noise(x0, eps, t, schedule: NoiseSchedule):
    """Noise clean data directly to level t using the closed form."""
    alpha_bar_t = schedule.alpha_bar[t]
    while alpha_bar_t.ndim < x0.ndim:
        alpha_bar_t = alpha_bar_t[..., None]
    return jnp.sqrt(alpha_bar_t) * x0 + jnp.sqrt(1.0 - alpha_bar_t) * eps


def forward_step(x, z, beta):
    """One forward step: shrink, then add Gaussian noise of variance beta."""
    return jnp.sqrt(1.0 - beta) * x + jnp.sqrt(beta) * z


def forward_chain(key, x0, schedule: NoiseSchedule):
    """Diffuse the same points from t = 0 through t = T.

    Returns states with shape (T + 1, n, d).
    """

    def step(x, inputs):
        beta, noise_key = inputs
        z = jax.random.normal(noise_key, x.shape, dtype=x.dtype)
        x_next = forward_step(x, z, beta)
        return x_next, x_next

    noise_keys = jax.random.split(key, schedule.betas.shape[0])
    _, later = jax.lax.scan(step, x0, (schedule.betas, noise_keys))
    return jnp.concatenate([x0[None, ...], later], axis=0)
