"""Deterministic DDIM and the mixture's exact conditional-mean noise."""

import jax
import jax.numpy as jnp
import numpy as np

from ddim import ddim_sample, ddim_step, exact_noise
from mlp import init_mlp, predict_noise
from schedule import add_noise, make_schedule


def test_ddim_step_to_zero_inverts_the_true_noise():
    schedule = make_schedule()
    clean = jnp.array([[3.0, 0.1], [-0.2, 2.7]], dtype=jnp.float32)
    eps = jnp.array([[0.4, -0.2], [0.1, 0.3]], dtype=jnp.float32)
    noisy = add_noise(clean, eps, t=40, schedule=schedule)
    recovered = ddim_step(noisy, eps, t=40, t_next=0, schedule=schedule)
    np.testing.assert_allclose(recovered, clean, rtol=1e-5, atol=1e-5)


def test_ddim_sample_is_deterministic():
    schedule = make_schedule(num_steps=8)
    params = init_mlp(jax.random.PRNGKey(3), (34, 8, 2))
    x_t = jax.random.normal(jax.random.PRNGKey(4), (5, 2), dtype=jnp.float32)
    levels = jnp.arange(8, -1, -1)

    def predict(state, level):
        return predict_noise(params, state, level)

    first = ddim_sample(predict, schedule, x_t, levels)
    second = ddim_sample(predict, schedule, x_t, levels)
    np.testing.assert_allclose(first, second, rtol=0, atol=0)


def test_exact_noise_is_zero_at_the_origin_and_tracks_standard_noise():
    schedule = make_schedule()
    origin = jnp.zeros((1, 2), dtype=jnp.float32)
    for level in (1, 50, 400, 1000):
        prediction = exact_noise(origin, level, schedule)
        np.testing.assert_allclose(prediction, 0.0, atol=1e-5)

    cloud = jax.random.normal(jax.random.PRNGKey(5), (32, 2), dtype=jnp.float32)
    prediction = np.asarray(exact_noise(cloud, 1000, schedule))
    np.testing.assert_allclose(prediction, cloud, rtol=1e-3, atol=2e-2)


def test_exact_noise_matches_an_independent_gaussian_mixture_calculation():
    schedule = make_schedule()
    level = 250
    points = np.array([[0.2, 2.5], [-2.8, 0.1], [0.0, 0.0], [1.5, -1.5]], dtype=np.float64)
    alpha_bar = float(schedule.alpha_bar[level])
    variance = 0.09
    scale = np.sqrt(alpha_bar)
    sigma = np.sqrt(max(1.0 - alpha_bar, 0.0))
    sigma2 = alpha_bar * variance + (1.0 - alpha_bar)
    centres = np.array([[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]])
    expected = []
    for point in points:
        delta = point[None, :] - scale * centres
        logits = -0.5 * np.sum(delta**2, axis=-1) / sigma2
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()
        coeff = scale * variance / sigma2
        clean = centres + coeff * delta
        component_noise = (point[None, :] - scale * clean) / sigma
        expected.append(weights @ component_noise)
    got = np.asarray(exact_noise(jnp.asarray(points, dtype=jnp.float32), level, schedule), dtype=np.float64)
    np.testing.assert_allclose(got, np.stack(expected), rtol=1e-4, atol=1e-5)
