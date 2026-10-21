"""Deterministic DDIM updates and the mixture's exact noise conditional mean."""

import jax
import jax.numpy as jnp

from mixture import CENTERS
from schedule import NoiseSchedule

COMPONENT_VARIANCE = 0.09


def ddim_step(x, eps_hat, t, t_next, schedule: NoiseSchedule):
    """Move a noisy point from level t to level t_next without drawing noise."""
    alpha_bar = schedule.alpha_bar[t]
    alpha_bar_next = schedule.alpha_bar[t_next]
    clean = (x - jnp.sqrt(1.0 - alpha_bar) * eps_hat) / jnp.sqrt(alpha_bar)
    return jnp.sqrt(alpha_bar_next) * clean + jnp.sqrt(1.0 - alpha_bar_next) * eps_hat


def ddim_sample(predict, schedule: NoiseSchedule, x_t, levels, return_trace=False):
    """Follow a decreasing level grid. ``levels[0]`` is the level of ``x_t``."""
    levels = jnp.asarray(levels)

    def step(state, pair):
        level, level_next = pair
        eps_hat = predict(state, jnp.full((state.shape[0],), level))
        updated = ddim_step(state, eps_hat, level, level_next, schedule)
        return updated, updated

    pairs = jnp.stack([levels[:-1], levels[1:]], axis=1)
    _final, arrived = jax.lax.scan(step, x_t, pairs)
    sample = arrived[-1]
    if return_trace:
        trace = jnp.concatenate([x_t[None, ...], arrived], axis=0)
        return sample, trace
    return sample


def exact_noise(points, t, schedule: NoiseSchedule, variance=COMPONENT_VARIANCE):
    """E[eps | x_t] for the four-cluster mixture, averaged over components."""
    alpha_bar = schedule.alpha_bar[t]
    scale = jnp.sqrt(alpha_bar)
    residual_scale = jnp.sqrt(jnp.maximum(1.0 - alpha_bar, 0.0))
    sigma2 = alpha_bar * variance + (1.0 - alpha_bar)
    shrunk = scale * CENTERS
    delta = points[:, None, :] - shrunk[None, :, :]
    logits = -0.5 * jnp.sum(delta**2, axis=-1) / sigma2
    weights = jax.nn.softmax(logits, axis=-1)
    coeff = scale * variance / sigma2
    clean = CENTERS[None, :, :] + coeff * delta
    component_noise = (points[:, None, :] - scale * clean) / residual_scale
    return jnp.sum(weights[:, :, None] * component_noise, axis=1)


def full_levels(num_steps=1000):
    """Every integer level from ``num_steps`` down to 0, inclusive."""
    return jnp.arange(num_steps, -1, -1)
