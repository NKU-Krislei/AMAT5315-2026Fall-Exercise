"""Equally weighted four-cluster Gaussian mixture."""

import jax
import jax.numpy as jnp

CENTERS = jnp.asarray(
    [
        [3.0, 0.0],
        [0.0, 3.0],
        [-3.0, 0.0],
        [0.0, -3.0],
    ],
    dtype=jnp.float32,
)
COMPONENT_STD = jnp.float32(0.3)


def sample_mixture(key, n: int, return_components: bool = False):
    """Draw n exact samples, x = mu_k + 0.3 z."""
    key_component, key_noise = jax.random.split(key)
    components = jax.random.randint(key_component, (n,), 0, CENTERS.shape[0])
    noise = jax.random.normal(key_noise, (n, CENTERS.shape[1]), dtype=CENTERS.dtype)
    samples = CENTERS[components] + COMPONENT_STD * noise
    if return_components:
        return samples, components
    return samples
