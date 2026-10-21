"""Equally weighted four-cluster Gaussian mixture."""

import jax
import jax.numpy as jnp
import numpy as np

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


def cluster_statistics(samples, centers=None):
    """Assign each point to the nearest target centre and summarise that group."""
    points = np.asarray(samples, dtype=np.float64)
    centres = np.asarray(CENTERS if centers is None else centers, dtype=np.float64)
    distances = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=-1)
    labels = distances.argmin(axis=1)
    rows = []
    for index, centre in enumerate(centres):
        group = points[labels == index]
        rows.append(
            {
                "index": index,
                "count": int(group.shape[0]),
                "fraction": float(group.shape[0] / points.shape[0]),
                "centre": group.mean(axis=0) if group.size else np.full(2, np.nan),
                "spread": group.std(axis=0) if group.size else np.full(2, np.nan),
            }
        )
    return labels, rows


def sample_mixture(key, n: int, return_components: bool = False):
    """Draw n exact samples, x = mu_k + 0.3 z."""
    key_component, key_noise = jax.random.split(key)
    components = jax.random.randint(key_component, (n,), 0, CENTERS.shape[0])
    noise = jax.random.normal(key_noise, (n, CENTERS.shape[1]), dtype=CENTERS.dtype)
    samples = CENTERS[components] + COMPONENT_STD * noise
    if return_components:
        return samples, components
    return samples
