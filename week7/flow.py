"""Conditional flow matching on straight paths from data to Gaussian noise."""

import jax
import jax.numpy as jnp

from mixture import CENTERS

COMPONENT_VARIANCE = 0.09


def flow_path(clean, noise, time):
    """Position and velocity on x_s = (1 - s) x_0 + s z."""
    time = jnp.asarray(time, dtype=clean.dtype).reshape((-1, 1))
    position = (1.0 - time) * clean + time * noise
    velocity = noise - clean
    return position, velocity


def exact_velocity(points, time, variance=COMPONENT_VARIANCE):
    """Marginal velocity E[z - x_0 | x_s = points, s = time]."""
    time = jnp.asarray(time, dtype=points.dtype)
    data_scale = 1.0 - time
    noise_scale = time
    sigma2 = data_scale**2 * variance + noise_scale**2
    delta = points[:, None, :] - (data_scale * CENTERS)[None, :, :]
    weights = jax.nn.softmax(-0.5 * jnp.sum(delta**2, axis=-1) / sigma2, axis=-1)
    expected_clean = CENTERS[None, :, :] + (data_scale * variance / sigma2) * delta
    expected_noise = (noise_scale / sigma2) * delta
    component_velocity = expected_noise - expected_clean
    return jnp.sum(weights[:, :, None] * component_velocity, axis=1)


def euler_sample(predict, x_t, times, return_trace=False):
    """Integrate dx/ds = v backward from time 1 toward time 0."""
    times = jnp.asarray(times)

    def step(state, pair):
        time, time_next = pair
        velocity = predict(state, jnp.full((state.shape[0],), time))
        updated = state - (time - time_next) * velocity
        return updated, updated

    pairs = jnp.stack([times[:-1], times[1:]], axis=1)
    _final, arrived = jax.lax.scan(step, x_t, pairs)
    sample = arrived[-1]
    if return_trace:
        return sample, jnp.concatenate([x_t[None, ...], arrived], axis=0)
    return sample


def time_grid(n_steps):
    """S equal intervals from s = 1 down to s = 0."""
    return jnp.linspace(1.0, 0.0, n_steps + 1, dtype=jnp.float32)
