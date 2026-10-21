"""Train the flow-matching velocity network from flow.toml."""

import tomllib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from flow import flow_path
from mixture import sample_mixture
from mlp import init_mlp, predict_noise
from train_diffusion import cosine_learning_rate, save_params

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "flow.toml"


def _read_config(config_path):
    return tomllib.loads(Path(config_path).read_text())


def train_velocity_predictor(
    updates=None,
    batch_size=None,
    hidden_widths=None,
    seed=None,
    checkpoint_dir=None,
    chunk_size=50,
    config_path=DEFAULT_CONFIG,
):
    config = _read_config(config_path)
    model = config["model"]
    training = config["training"]
    updates = int(training["updates"] if updates is None else updates)
    batch_size = int(training["batch_size"] if batch_size is None else batch_size)
    hidden_widths = tuple(model["hidden_widths"] if hidden_widths is None else hidden_widths)
    seed = int(training["seed"] if seed is None else seed)
    n_freq = int(model["time_features"]) // 2
    frequency_base = float(model["time_frequency_base"])
    time_scale = float(model["time_scale"])
    learning_rate = float(training["learning_rate"])
    final_learning_rate = float(training["final_learning_rate"])
    checkpoint_dir = Path(checkpoint_dir or ROOT / "artifacts" / "flow")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    layer_sizes = (2 + 2 * n_freq, *hidden_widths, 2)
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = init_mlp(init_key, layer_sizes)
    optimizer = optax.adam(
        lambda count: cosine_learning_rate(count, updates, learning_rate, final_learning_rate),
        b1=float(training["beta1"]),
        b2=float(training["beta2"]),
        eps=float(training["epsilon"]),
    )
    opt_state = optimizer.init(params)

    def predict(params, points, times):
        return predict_noise(
            params,
            points,
            times,
            scale=time_scale,
            offset=0.0,
            n_freq=n_freq,
            base=frequency_base,
        )

    def make_chunk(n_steps):
        @jax.jit
        def chunk(params, opt_state, key):
            def body(carry, _):
                params, opt_state, key = carry
                key, data_key, noise_key, time_key = jax.random.split(key, 4)
                clean = sample_mixture(data_key, batch_size)
                noise = jax.random.normal(noise_key, clean.shape, dtype=clean.dtype)
                time = jax.random.uniform(time_key, (batch_size,), dtype=clean.dtype)
                position, velocity = flow_path(clean, noise, time)

                def loss_fn(current):
                    residual = predict(current, position, time) - velocity
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

    save_params(checkpoint_dir / "checkpoint_00000.npz", params)
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
        print(f"flow update {completed}/{updates}  loss {chunk_loss.mean():.6f}", flush=True)

    save_params(checkpoint_dir / f"checkpoint_{completed:05d}.npz", params)
    loss = np.concatenate(losses)
    np.save(checkpoint_dir / "loss.npy", loss)
    return {"loss": loss, "params": params}


def main():
    train_velocity_predictor()


if __name__ == "__main__":
    main()
