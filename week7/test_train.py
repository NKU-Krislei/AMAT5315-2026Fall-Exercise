"""Short training loop for the noise predictor. The 25k run is not a unit test."""

import jax
import jax.numpy as jnp

from train_diffusion import cosine_learning_rate, train_noise_predictor


def test_cosine_learning_rate_runs_from_the_initial_rate_to_zero():
    assert cosine_learning_rate(0, updates=25000, initial=1e-3, final=0.0) == jnp.float32(1e-3)
    assert cosine_learning_rate(25000, updates=25000, initial=1e-3, final=0.0) == 0.0
    mid = float(cosine_learning_rate(12500, updates=25000, initial=1e-3, final=0.0))
    assert 4e-4 < mid < 6e-4


def test_a_few_updates_record_loss_and_keep_the_initialization_checkpoint(tmp_path):
    result = train_noise_predictor(
        updates=4,
        batch_size=8,
        hidden_widths=(8, 8),
        seed=2026,
        checkpoint_updates=(0, 2, 4),
        checkpoint_dir=tmp_path,
        chunk_size=2,
    )
    assert result["loss"].shape == (4,)
    assert jnp.all(jnp.isfinite(result["loss"]))
    assert (tmp_path / "checkpoint_00000.npz").exists()
    assert (tmp_path / "checkpoint_00002.npz").exists()
    assert (tmp_path / "checkpoint_00004.npz").exists()
    init = result["checkpoints"][0]
    trained = result["checkpoints"][4]
    assert any(not jnp.allclose(a, b) for a, b in zip(jax.tree.leaves(init), jax.tree.leaves(trained)))
