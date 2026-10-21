"""Target sampling and forward diffusion for the four-cluster mixture."""

import tomllib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mixture import CENTERS, cluster_statistics, sample_mixture
from schedule import add_noise, forward_chain, forward_step, make_schedule

ROOT = Path(__file__).resolve().parent
EXPECTED_ALPHA_BAR_T = 4.0358297653756754e-05


def test_diffusion_toml_matches_sheet():
    data = tomllib.loads((ROOT / "diffusion.toml").read_text())
    assert data["model"]["hidden_widths"] == [128, 128, 128]
    assert data["model"]["activation"] == "SiLU"
    assert data["model"]["time_features"] == 32
    assert data["model"]["time_frequency_base"] == 10_000.0
    assert data["model"]["time_offset"] == -1
    assert data["model"]["bias_initialization"] == 0.0
    assert data["training"]["updates"] == 25_000
    assert data["training"]["batch_size"] == 512
    assert data["training"]["optimizer"] == "Adam"
    assert data["training"]["beta1"] == 0.9
    assert data["training"]["beta2"] == 0.999
    assert data["training"]["epsilon"] == 1e-8
    assert data["training"]["learning_rate"] == 0.001
    assert data["training"]["learning_rate_decay"] == "cosine"
    assert data["training"]["final_learning_rate"] == 0.0
    assert data["training"]["precision"] == "float32"
    assert data["training"]["seed"] == 2026
    assert data["training"]["checkpoint_updates"] == [
        0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000,
    ]


def test_beta_schedule_endpoints_and_cumulative_product():
    schedule = make_schedule()
    betas = np.asarray(schedule.betas)
    assert betas.shape == (1000,)
    np.testing.assert_allclose(betas[0], 1e-4, rtol=1e-6, atol=0)
    np.testing.assert_allclose(betas[-1], 0.02, rtol=1e-6, atol=0)
    assert np.all(np.diff(betas) > 0)
    np.testing.assert_allclose(np.asarray(schedule.alphas), 1 - betas, rtol=1e-6, atol=0)

    beta_exact = np.linspace(1e-4, 0.02, 1000, dtype=np.float64)
    alpha_bar = np.asarray(schedule.alpha_bar, dtype=np.float64)
    assert alpha_bar.shape == (1001,)
    np.testing.assert_allclose(alpha_bar[0], 1.0)
    np.testing.assert_allclose(alpha_bar[1:], np.cumprod(1 - beta_exact), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(alpha_bar[-1], EXPECTED_ALPHA_BAR_T, rtol=1e-5, atol=0)


def test_direct_noising_matches_the_first_forward_step():
    schedule = make_schedule()
    x0 = jnp.array([[1.0, -2.0], [0.5, 0.25]], dtype=jnp.float32)
    eps = jnp.array([[0.1, -0.2], [0.3, 0.4]], dtype=jnp.float32)
    direct = add_noise(x0, eps, t=1, schedule=schedule)
    stepped = forward_step(x0, eps, schedule.betas[0])
    np.testing.assert_allclose(direct, stepped, rtol=1e-5, atol=1e-6)


def test_forward_chain_follows_the_same_points_to_a_standard_gaussian():
    schedule = make_schedule()
    key = jax.random.PRNGKey(2026)
    key_data, key_chain = jax.random.split(key)
    x0 = sample_mixture(key_data, 4000)
    states = forward_chain(key_chain, x0, schedule)
    assert states.shape == (1001, 4000, 2)
    np.testing.assert_allclose(states[0], x0, rtol=1e-5, atol=1e-6)

    final = np.asarray(states[-1])
    mean = final.mean(axis=0)
    var = final.var(axis=0)
    assert np.all(np.abs(mean) < 0.1)
    assert np.all(np.abs(var - 1.0) < 0.1)


def test_mixture_has_equal_populations_centres_and_width():
    x, components = sample_mixture(jax.random.PRNGKey(7), 20_000, return_components=True)
    points = np.asarray(x)
    labels = np.asarray(components)
    centres = np.asarray(CENTERS)
    assert points.shape == (20_000, 2)
    assert centres.shape == (4, 2)
    for index, centre in enumerate(centres):
        group = points[labels == index]
        fraction = group.shape[0] / points.shape[0]
        assert abs(fraction - 0.25) < 0.02
        np.testing.assert_allclose(group.mean(axis=0), centre, atol=0.03)
        np.testing.assert_allclose(group.std(axis=0), 0.3, atol=0.02)


def test_nearest_centre_statistics_match_the_sampled_components():
    points, components = sample_mixture(jax.random.PRNGKey(11), 20_000, return_components=True)
    labels, rows = cluster_statistics(points)
    assert np.mean(labels == np.asarray(components)) > 0.99
    for row, centre in zip(rows, np.asarray(CENTERS)):
        assert abs(row["fraction"] - 0.25) < 0.02
        np.testing.assert_allclose(row["centre"], centre, atol=0.03)
        np.testing.assert_allclose(row["spread"], 0.3, atol=0.02)
