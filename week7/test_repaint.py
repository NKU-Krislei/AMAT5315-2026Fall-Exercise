import jax
import jax.numpy as jnp
import numpy as np

from repaint import repaint, repaint_jump
from schedule import make_schedule


def test_last_jump_keeps_observations_and_clips_the_clean_estimate():
    state = jnp.array([0.4, 3.0], dtype=jnp.float32)
    observation = jnp.array([0.25, 0.0], dtype=jnp.float32)
    mask = jnp.array([1.0, 0.0], dtype=jnp.float32)
    got = repaint_jump(
        state,
        observation,
        mask,
        jnp.zeros_like(state),
        jnp.zeros_like(state),
        jnp.ones_like(state),
        jnp.float32(0.25),
        jnp.float32(1.0),
    )
    np.testing.assert_allclose(got, np.array([0.25, 1.0]), atol=1e-6)


def test_intermediate_jump_uses_the_posterior_and_the_noised_observation():
    state = jnp.full((2,), 0.2, dtype=jnp.float32)
    observation = jnp.array([0.2, 0.0], dtype=jnp.float32)
    mask = jnp.array([1.0, 0.0], dtype=jnp.float32)
    got = repaint_jump(
        state,
        observation,
        mask,
        jnp.zeros_like(state),
        jnp.zeros_like(state),
        jnp.zeros_like(state),
        jnp.float32(0.16),
        jnp.float32(0.64),
    )
    np.testing.assert_allclose(got, np.array([0.16, 0.4]), rtol=1e-5, atol=1e-5)


def test_repaint_retries_each_jump_except_the_one_to_zero():
    seen = []

    def predict(state, level):
        seen.append((int(level), np.array(state)))
        return jnp.zeros_like(state)

    observation = jnp.array([0.3, -0.2], dtype=jnp.float32)
    mask = jnp.array([1.0, 0.0], dtype=jnp.float32)
    done = repaint(
        predict,
        make_schedule(),
        observation,
        mask,
        jax.random.PRNGKey(7),
        jnp.array([40, 20, 0]),
        attempts=2,
    )
    assert [level for level, _state in seen] == [40, 40, 20]
    assert not np.allclose(seen[0][1], seen[1][1])
    np.testing.assert_allclose(done[0], 0.3, atol=1e-6)


def test_batched_seeds_keep_observations_and_differ_inside_the_hole():
    def predict(state, level):
        return jnp.zeros_like(state)

    observation = jnp.array([[0.3, -0.2], [0.3, -0.2]], dtype=jnp.float32)
    mask = jnp.array([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
    keys = jnp.stack([jax.random.PRNGKey(1), jax.random.PRNGKey(2)])
    done = repaint(
        predict,
        make_schedule(),
        observation,
        mask,
        keys,
        jnp.array([40, 20, 0]),
        attempts=2,
    )
    np.testing.assert_allclose(done[:, 0], np.array([0.3, 0.3]), atol=1e-6)
    assert not np.allclose(done[0, 1], done[1, 1])
