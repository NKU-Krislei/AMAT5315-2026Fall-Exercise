import numpy as np

from unet import timestep_embedding


def test_timestep_embedding_uses_sine_then_cosine_and_the_shifted_frequencies():
    half = 64
    exponent = -np.log(10000.0) * np.arange(half) / (half - 1.0)
    angles = 7.0 * np.exp(exponent)
    expected = np.concatenate([np.sin(angles), np.cos(angles)])
    got = np.asarray(timestep_embedding(np.array([7.0], dtype=np.float32)))[0]
    assert np.allclose(got, expected, rtol=1e-5, atol=1e-5)
