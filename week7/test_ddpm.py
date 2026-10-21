"""Noise predictor and DDPM reverse steps."""

import jax
import jax.numpy as jnp
import numpy as np

from ddpm import ddpm_mean, ddpm_sample
from mlp import init_mlp, predict_noise, silu, time_features
from schedule import make_schedule


def reference_time_features(levels, scale=1.0, offset=-1.0, n_freq=16, base=10_000.0):
    levels = np.asarray(levels, dtype=np.float64)
    frequencies = np.arange(n_freq, dtype=np.float64)
    omega = base ** (-frequencies / n_freq)
    angles = (scale * levels[:, None] + offset) * omega[None, :]
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)


def test_time_features_use_the_sheet_frequencies_and_offset():
    levels = np.array([1, 2, 1000], dtype=np.float32)
    got = np.asarray(time_features(jnp.asarray(levels), scale=1.0, offset=-1.0))
    # Large angles at t=1000 are not exact in float32, so allow that rounding.
    np.testing.assert_allclose(got, reference_time_features(levels), rtol=1e-4, atol=2e-5)
    np.testing.assert_allclose(got[0, :16], 0.0, atol=1e-6)
    np.testing.assert_allclose(got[0, 16:], 1.0, atol=1e-6)


def test_silu_matches_the_swish_formula():
    values = jnp.linspace(-4, 4, 9)
    np.testing.assert_allclose(silu(values), values / (1 + jnp.exp(-values)), rtol=1e-5, atol=1e-6)


def test_initial_weights_use_fan_in_std_and_zero_bias():
    key = jax.random.PRNGKey(2026)
    params = init_mlp(key, (34, 128, 128, 128, 2))
    for fan_in, (weight, bias) in zip((34, 128, 128, 128), params):
        assert weight.shape == (fan_in, weight.shape[1])
        np.testing.assert_allclose(bias, 0.0)
        np.testing.assert_allclose(np.std(np.asarray(weight)), 1 / np.sqrt(fan_in), rtol=0.05, atol=0)


def test_predictor_reads_the_noisy_point_and_its_level():
    params = init_mlp(jax.random.PRNGKey(0), (34, 16, 2))
    x = jnp.array([[0.2, -0.4], [1.0, 0.5]], dtype=jnp.float32)
    early = predict_noise(params, x, jnp.array([1, 1]))
    late = predict_noise(params, x, jnp.array([1000, 1000]))
    assert early.shape == (2, 2)
    assert not np.allclose(early, late)


def test_ddpm_mean_matches_the_reverse_formula():
    schedule = make_schedule(num_steps=8)
    x = jnp.array([[0.4, -1.2], [0.0, 0.7]], dtype=jnp.float32)
    eps = jnp.array([[0.2, 0.1], [-0.3, 0.5]], dtype=jnp.float32)
    t = 6
    beta = schedule.betas[t - 1]
    alpha = schedule.alphas[t - 1]
    alpha_bar = schedule.alpha_bar[t]
    expected = (1 / jnp.sqrt(alpha)) * (x - beta / jnp.sqrt(1 - alpha_bar) * eps)
    np.testing.assert_allclose(ddpm_mean(x, eps, t, schedule), expected, rtol=1e-5, atol=1e-6)


def test_ddpm_sample_adds_noise_except_on_the_final_step():
    schedule = make_schedule(num_steps=4)
    x_t = jnp.array([[0.5, -0.25]], dtype=jnp.float32)
    step_noise = jnp.ones((5, 1, 2), dtype=jnp.float32)

    def predict_zero(x, _level):
        return jnp.zeros_like(x)

    sample, trace = ddpm_sample(predict_zero, schedule, x_t, step_noise, return_trace=True)
    assert sample.shape == (1, 2)
    assert trace.shape == (5, 1, 2)
    np.testing.assert_allclose(trace[-1], x_t)

    state = x_t
    for level in range(4, 0, -1):
        mean = ddpm_mean(state, jnp.zeros_like(state), level, schedule)
        noise = jnp.zeros_like(state) if level == 1 else step_noise[level]
        state = mean + jnp.sqrt(schedule.betas[level - 1]) * noise
        np.testing.assert_allclose(trace[level - 1], state, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(sample, state, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(trace[0], ddpm_mean(trace[1], jnp.zeros_like(trace[1]), 1, schedule))
