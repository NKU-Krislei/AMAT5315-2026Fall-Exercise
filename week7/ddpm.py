"""DDPM reverse step. The final step adds no fresh noise."""

import jax
import jax.numpy as jnp

from schedule import NoiseSchedule


def ddpm_mean(x, eps_hat, t, schedule: NoiseSchedule):
    """Conditional-mean estimate of the previous state at level t - 1."""
    beta = schedule.betas[t - 1]
    alpha = schedule.alphas[t - 1]
    alpha_bar = schedule.alpha_bar[t]
    return (1.0 / jnp.sqrt(alpha)) * (x - (beta / jnp.sqrt(1.0 - alpha_bar)) * eps_hat)


def ddpm_sample(predict, schedule: NoiseSchedule, x_t, step_noise, return_trace=False):
    """Run DDPM from level T down to 0.

    ``x_t`` is the state at level T. ``step_noise[t]`` is the standard Gaussian
    draw used when leaving level t, for t > 1. The returned trace is indexed by
    noise level, so ``trace[0]`` is the clean sample and ``trace[T]`` is ``x_t``.
    """
    num_steps = int(schedule.betas.shape[0])

    def step(state, level):
        eps_hat = predict(state, jnp.full((state.shape[0],), level))
        mean = ddpm_mean(state, eps_hat, level, schedule)
        noise = jnp.where(level > 1, step_noise[level], jnp.zeros_like(state))
        previous = mean + jnp.sqrt(schedule.betas[level - 1]) * noise
        return previous, previous

    levels = jnp.arange(num_steps, 0, -1)
    _final, arrived = jax.lax.scan(step, x_t, levels)
    # arrived[0] is the state at level T - 1; the last entry is level 0.
    stacked = jnp.concatenate([x_t[None, ...], arrived], axis=0)
    trace = jnp.flip(stacked, axis=0)
    sample = trace[0]
    if return_trace:
        return sample, trace
    return sample
