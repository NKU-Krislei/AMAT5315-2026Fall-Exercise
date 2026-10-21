"""RePaint on a coarse DDPM grid. Only the clean Tweedie estimate is clipped."""

import jax
import jax.numpy as jnp
import numpy as np

from schedule import NoiseSchedule


def repaint_jump(
    state,
    observation,
    mask,
    eps_hat,
    xi,
    eta,
    alpha_bar,
    alpha_bar_next,
    low=-1.0,
    high=1.0,
):
    """One reverse jump, then overwrite observed entries with noised observations.

    ``alpha_bar`` is the cumulative product at the current level. The variance
    is the coarse posterior variance, which is zero at the final level.
    """
    ratio = alpha_bar / alpha_bar_next
    clean = (state - jnp.sqrt(1.0 - alpha_bar) * eps_hat) / jnp.sqrt(alpha_bar)
    clean = jnp.clip(clean, low, high)
    remaining = 1.0 - alpha_bar
    mean = (
        jnp.sqrt(alpha_bar_next) * (1.0 - ratio) / remaining * clean
        + jnp.sqrt(ratio) * (1.0 - alpha_bar_next) / remaining * state
    )
    variance = (1.0 - alpha_bar_next) * (1.0 - ratio) / remaining
    proposal = mean + jnp.sqrt(jnp.maximum(variance, 0.0)) * xi
    noised_observation = jnp.sqrt(alpha_bar_next) * observation + jnp.sqrt(1.0 - alpha_bar_next) * eta
    return mask * noised_observation + (1.0 - mask) * proposal


def _split(key, count):
    if key.ndim == 1:
        parts = jax.random.split(key, count)
        return [parts[index] for index in range(count)]
    parts = jax.vmap(lambda item: jax.random.split(item, count))(key)
    return [parts[:, index] for index in range(count)]


def _normal_like(key, like):
    if key.ndim == 1:
        return jax.random.normal(key, like.shape, dtype=jnp.float32)
    tail = like.shape[1:]
    return jax.vmap(lambda item: jax.random.normal(item, tail, dtype=jnp.float32))(key)


def repaint(
    predict,
    schedule: NoiseSchedule,
    observation,
    mask,
    key,
    levels,
    attempts=2,
    low=-1.0,
    high=1.0,
):
    """Inpaint from ``levels[0]`` down to 0.

    ``mask`` is 1 on observed entries. A key of shape (batch, 2) runs that many
    independent seeds on a matching batch of observations. Every jump except
    the last is repeated ``attempts`` times, with forward noise between repeats.
    ``predict(state, level)`` receives the mathematical noise level.
    """
    levels = np.asarray(levels, dtype=np.int32)
    observation = jnp.asarray(observation, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    key, start_key = _split(key, 2)
    state = _normal_like(start_key, observation)

    @jax.jit
    def jump(current, eps_hat, xi, eta, alpha_bar, alpha_bar_next):
        return repaint_jump(
            current,
            observation,
            mask,
            eps_hat,
            xi,
            eta,
            alpha_bar,
            alpha_bar_next,
            low,
            high,
        )

    for level, level_next in zip(levels[:-1], levels[1:]):
        repeats = 1 if int(level_next) == 0 else attempts
        alpha_bar = schedule.alpha_bar[int(level)]
        alpha_bar_next = schedule.alpha_bar[int(level_next)]
        ratio = alpha_bar / alpha_bar_next
        for attempt in range(repeats):
            if attempt:
                key, fresh_key = _split(key, 2)
                fresh = _normal_like(fresh_key, state)
                state = jnp.sqrt(ratio) * state + jnp.sqrt(jnp.maximum(1.0 - ratio, 0.0)) * fresh
            key, xi_key, eta_key = _split(key, 3)
            xi = _normal_like(xi_key, state)
            eta = _normal_like(eta_key, state)
            eps_hat = predict(state, jnp.int32(level))
            state = jump(state, eps_hat, xi, eta, alpha_bar, alpha_bar_next)
    return state


def repaint_levels(start=1000, stride=20):
    """Levels 1000, 980, ..., 20, 0."""
    return np.concatenate([np.arange(start, 0, -stride, dtype=np.int32), np.zeros(1, dtype=np.int32)])
