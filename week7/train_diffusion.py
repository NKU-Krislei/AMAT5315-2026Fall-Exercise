"""Train the time-conditioned noise predictor from diffusion.toml."""

import tomllib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mixture import sample_mixture
from mlp import init_mlp, predict_noise
from schedule import add_noise, make_schedule

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "diffusion.toml"


def cosine_learning_rate(count, updates, initial=1e-3, final=0.0):
    """Cosine decay from ``initial`` at update 0 to ``final`` at ``updates``."""
    count = jnp.asarray(count, dtype=jnp.float32)
    steps = jnp.asarray(updates, dtype=jnp.float32)
    initial = jnp.asarray(initial, dtype=jnp.float32)
    final = jnp.asarray(final, dtype=jnp.float32)
    fraction = jnp.minimum(count, steps) / steps
    decay = 0.5 * (1.0 + jnp.cos(jnp.pi * fraction))
    return final + (initial - final) * decay


def save_params(path, params):
    payload = {}
    for index, (weight, bias) in enumerate(params):
        payload[f"W{index}"] = np.asarray(weight)
        payload[f"b{index}"] = np.asarray(bias)
    np.savez(path, **payload)


def load_params(path):
    with np.load(path) as data:
        count = sum(name.startswith("W") for name in data.files)
        return tuple(
            (jnp.asarray(data[f"W{index}"]), jnp.asarray(data[f"b{index}"]))
            for index in range(count)
        )


def _read_config(config_path):
    return tomllib.loads(Path(config_path).read_text())


def train_noise_predictor(
    updates=None,
    batch_size=None,
    hidden_widths=None,
    seed=None,
    checkpoint_updates=None,
    checkpoint_dir=None,
    chunk_size=50,
    config_path=DEFAULT_CONFIG,
    learning_rate=None,
    final_learning_rate=None,
):
    config = _read_config(config_path)
    model = config["model"]
    training = config["training"]
    updates = int(training["updates"] if updates is None else updates)
    batch_size = int(training["batch_size"] if batch_size is None else batch_size)
    hidden_widths = tuple(model["hidden_widths"] if hidden_widths is None else hidden_widths)
    seed = int(training["seed"] if seed is None else seed)
    checkpoint_updates = tuple(
        training["checkpoint_updates"] if checkpoint_updates is None else checkpoint_updates
    )
    learning_rate = float(training["learning_rate"] if learning_rate is None else learning_rate)
    final_learning_rate = float(
        training["final_learning_rate"] if final_learning_rate is None else final_learning_rate
    )
    n_freq = int(model["time_features"]) // 2
    frequency_base = float(model["time_frequency_base"])
    time_offset = float(model["time_offset"])
    checkpoint_dir = Path(checkpoint_dir or ROOT / "artifacts" / "diffusion")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    schedule = make_schedule()
    layer_sizes = (2 + 2 * n_freq, *hidden_widths, 2)
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = init_mlp(init_key, layer_sizes)
    learning_rate_fn = lambda count: cosine_learning_rate(
        count, updates, learning_rate, final_learning_rate
    )
    optimizer = optax.adam(
        learning_rate_fn,
        b1=float(training["beta1"]),
        b2=float(training["beta2"]),
        eps=float(training["epsilon"]),
    )
    opt_state = optimizer.init(params)

    def predict(params, points, levels):
        return predict_noise(
            params,
            points,
            levels,
            scale=1.0,
            offset=time_offset,
            n_freq=n_freq,
            base=frequency_base,
        )

    def make_chunk(n_steps):
        @jax.jit
        def chunk(params, opt_state, key):
            def body(carry, _):
                params, opt_state, key = carry
                key, data_key, noise_key, level_key = jax.random.split(key, 4)
                clean = sample_mixture(data_key, batch_size)
                eps = jax.random.normal(noise_key, clean.shape, dtype=clean.dtype)
                levels = jax.random.randint(level_key, (batch_size,), 1, schedule.betas.shape[0] + 1)
                noisy = add_noise(clean, eps, levels, schedule)

                def loss_fn(current):
                    residual = predict(current, noisy, levels) - eps
                    return jnp.mean(residual**2)

                loss, grads = jax.value_and_grad(loss_fn)(params)
                opt_updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, opt_updates)
                return (params, opt_state, key), loss

            (params, opt_state, key), loss = jax.lax.scan(
                body, (params, opt_state, key), None, length=n_steps
            )
            return params, opt_state, key, loss

        return chunk

    checkpoint_set = set(int(step) for step in checkpoint_updates)
    stored = {}
    if 0 in checkpoint_set:
        save_params(checkpoint_dir / "checkpoint_00000.npz", params)
        stored[0] = params

    losses = []
    completed = 0
    chunk_fn = None
    active_chunk = None
    while completed < updates:
        n_steps = min(chunk_size, updates - completed)
        if n_steps != active_chunk:
            chunk_fn = make_chunk(n_steps)
            active_chunk = n_steps
        params, opt_state, key, chunk_loss = chunk_fn(params, opt_state, key)
        chunk_loss = np.asarray(chunk_loss)
        losses.append(chunk_loss)
        completed += n_steps
        print(f"update {completed}/{updates}  loss {chunk_loss.mean():.6f}", flush=True)
        if completed in checkpoint_set:
            save_params(checkpoint_dir / f"checkpoint_{completed:05d}.npz", params)
            stored[completed] = params

    loss = np.concatenate(losses)
    np.save(checkpoint_dir / "loss.npy", loss)
    return {"loss": loss, "checkpoints": stored}


def main():
    train_noise_predictor()


if __name__ == "__main__":
    main()
