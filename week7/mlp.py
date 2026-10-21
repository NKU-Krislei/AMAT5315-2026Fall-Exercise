"""Time-conditioned SiLU network used as a noise or velocity predictor."""

import jax
import jax.numpy as jnp


def silu(values):
    return values / (1.0 + jnp.exp(-values))


def time_features(levels, scale=1.0, offset=-1.0, n_freq=16, base=10_000.0):
    """Sine and cosine features. Diffusion uses scale 1 and offset -1."""
    levels = jnp.asarray(levels, dtype=jnp.float32)
    frequencies = jnp.arange(n_freq, dtype=jnp.float32)
    omega = jnp.asarray(base, dtype=jnp.float32) ** (-frequencies / n_freq)
    angles = (scale * levels[:, None] + offset) * omega[None, :]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


def init_mlp(key, layer_sizes):
    """Normal weights with std 1/sqrt(fan-in) and zero biases."""
    params = []
    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        key, subkey = jax.random.split(key)
        std = 1.0 / jnp.sqrt(jnp.asarray(fan_in, dtype=jnp.float32))
        weight = jax.random.normal(subkey, (fan_in, fan_out), dtype=jnp.float32) * std
        bias = jnp.zeros((fan_out,), dtype=jnp.float32)
        params.append((weight, bias))
    return tuple(params)


def mlp_apply(params, inputs):
    hidden = inputs
    last = len(params) - 1
    for index, (weight, bias) in enumerate(params):
        hidden = hidden @ weight + bias
        if index != last:
            hidden = silu(hidden)
    return hidden


def predict_noise(params, points, levels, scale=1.0, offset=-1.0, n_freq=16, base=10_000.0):
    features = time_features(levels, scale=scale, offset=offset, n_freq=n_freq, base=base)
    return mlp_apply(params, jnp.concatenate([points, features], axis=-1))
