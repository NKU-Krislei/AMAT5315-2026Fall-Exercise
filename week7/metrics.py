"""Energy distance and relative velocity error."""

import numpy as np


def energy_distance(generated, target):
    """Székely energy distance, averaged over distinct within-sample pairs."""
    generated = np.asarray(generated, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    between = np.linalg.norm(generated[:, None, :] - target[None, :, :], axis=-1).mean()
    return float(2.0 * between - _mean_distinct_pairs(generated) - _mean_distinct_pairs(target))


def _mean_distinct_pairs(points):
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    count = points.shape[0]
    return distances.sum() / (count * (count - 1))


def relative_rms(predicted, exact):
    predicted = np.asarray(predicted, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    return float(np.sqrt(np.sum((predicted - exact) ** 2) / np.sum(exact**2)))
