"""Flow matching paths, exact marginal velocity, and energy distance."""

import tomllib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from flow import euler_sample, exact_velocity, flow_path
from metrics import energy_distance
from mlp import time_features

ROOT = Path(__file__).resolve().parent


def test_flow_toml_uses_continuous_time_features_and_a_new_seed():
    data = tomllib.loads((ROOT / "flow.toml").read_text())
    assert data["model"]["time_scale"] == 1000.0
    assert data["training"]["seed"] == 2029
    assert data["training"]["updates"] == 25_000
    times = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32)
    got = np.asarray(time_features(times, scale=1000.0, offset=0.0))
    angles = 1000.0 * np.asarray(times)[:, None] * (10_000.0 ** (-np.arange(16) / 16.0))[None, :]
    expected = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=2e-5)


def test_straight_path_has_constant_velocity_from_data_to_noise():
    clean = jnp.array([[3.0, 0.0]], dtype=jnp.float32)
    noise = jnp.array([[0.0, -1.0]], dtype=jnp.float32)
    position, velocity = flow_path(clean, noise, jnp.array([0.25], dtype=jnp.float32))
    np.testing.assert_allclose(position, 0.75 * clean + 0.25 * noise)
    np.testing.assert_allclose(velocity, noise - clean)


def test_exact_velocity_at_pure_noise_is_the_position():
    points = jnp.array([[0.4, -1.2], [2.0, 0.3], [-0.5, 0.7]], dtype=jnp.float32)
    np.testing.assert_allclose(exact_velocity(points, 1.0), points, rtol=1e-4, atol=1e-5)


def test_euler_step_moves_opposite_the_forward_velocity():
    def predict(state, time):
        return jnp.broadcast_to(jnp.array([1.0, -2.0], dtype=jnp.float32), state.shape)

    start = jnp.array([[0.0, 0.0]], dtype=jnp.float32)
    times = jnp.array([1.0, 0.5, 0.0], dtype=jnp.float32)
    sample, trace = euler_sample(predict, start, times, return_trace=True)
    np.testing.assert_allclose(trace[0], start)
    np.testing.assert_allclose(trace[1], jnp.array([[-0.5, 1.0]]))
    np.testing.assert_allclose(sample, jnp.array([[-1.0, 2.0]]))
    again, _ = euler_sample(predict, start, times, return_trace=True)
    np.testing.assert_allclose(sample, again, rtol=0, atol=0)


def test_energy_distance_is_small_for_one_mixture_and_large_across_a_shift():
    from mixture import sample_mixture

    key = jax.random.PRNGKey(9)
    left_key, right_key, shifted_key = jax.random.split(key, 3)
    left = np.asarray(sample_mixture(left_key, 800))
    right = np.asarray(sample_mixture(right_key, 800))
    shifted = np.asarray(sample_mixture(shifted_key, 800)) + np.array([6.0, 0.0])
    assert energy_distance(left, right) < 0.05
    assert energy_distance(left, shifted) > 1.0
