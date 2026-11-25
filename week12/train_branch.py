"""Imitation learning of Strong Branching, following train_branch.design.toml.

The shared network is f(phi) = w2^T tanh(W1^T phi + b1) + b2, with W1 shaped
(12, 48). Weights use Glorot normal initialization from seed 2026. Adam uses
beta1 0.9, beta2 0.999 and epsilon 1e-8. Feature statistics are the population
mean and standard deviation over every training candidate row.
"""

import argparse
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


FEATURE_DIM = 12
HIDDEN = 48
EPOCHS = 120
BATCH = 64
LEARNING_RATE = 4e-3
L2 = 2e-4
BETA1 = 0.9
BETA2 = 0.999
ADAM_EPS = 1e-8
SEED = 2026


def load_split(path):
    with open(path) as handle:
        document = json.load(handle)
    train = document["train"]
    validation = document["validation"]
    train_ids = {int(node["instance"]) for node in train}
    valid_ids = {int(node["instance"]) for node in validation}
    overlap = sorted(train_ids & valid_ids)
    if overlap:
        raise SystemExit(f"training and validation share instances {overlap}")
    return train, validation, train_ids, valid_ids


def pack(nodes):
    if not nodes:
        raise SystemExit("a split has no nodes")
    width = max(len(node["candidates"]) for node in nodes)
    feature_rows = []
    raw = [[[0.0] * FEATURE_DIM for _ in range(width)] for _ in nodes]
    raw_mask = [[False] * width for _ in nodes]
    raw_targets = [[0.0] * width for _ in nodes]
    for row, node in enumerate(nodes):
        scores = [float(score) for score in node["scores"]]
        best = max(scores)
        tolerance = 1e-9 * max(1.0, abs(best))
        tied = [index for index, score in enumerate(scores) if score >= best - tolerance]
        share = 1.0 / len(tied)
        for index, candidate_features in enumerate(node["features"]):
            raw[row][index] = [float(value) for value in candidate_features]
            raw_mask[row][index] = True
            feature_rows.append(raw[row][index])
        for index in tied:
            raw_targets[row][index] = share
    return (
        jnp.asarray(raw, dtype=jnp.float64),
        jnp.asarray(raw_mask, dtype=bool),
        jnp.asarray(raw_targets, dtype=jnp.float64),
        jnp.asarray(feature_rows, dtype=jnp.float64),
    )


def standardize(features, mu, std):
    return (features - mu) / std


def init_params(key):
    key_w1, key_w2 = jax.random.split(key)
    w1 = jax.random.normal(key_w1, (FEATURE_DIM, HIDDEN), dtype=jnp.float64) * jnp.sqrt(
        2.0 / (FEATURE_DIM + HIDDEN)
    )
    b1 = jnp.zeros((HIDDEN,), dtype=jnp.float64)
    w2 = jax.random.normal(key_w2, (HIDDEN,), dtype=jnp.float64) * jnp.sqrt(2.0 / (HIDDEN + 1))
    b2 = jnp.array(0.0, dtype=jnp.float64)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def logits_of(params, features):
    hidden = jnp.tanh(features @ params["w1"] + params["b1"])
    return hidden @ params["w2"] + params["b2"]


def masked_log_softmax(logits, mask):
    masked = jnp.where(mask, logits, -1.0e30)
    shift = jnp.max(masked, axis=-1, keepdims=True)
    shifted = jnp.where(mask, logits - shift, -1.0e30)
    log_denominator = jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))
    return jnp.where(mask, logits - shift - log_denominator, 0.0)


def cross_entropy(params, features, mask, targets):
    log_probability = masked_log_softmax(logits_of(params, features), mask)
    per_node = -jnp.sum(targets * log_probability, axis=-1)
    return jnp.mean(per_node)


def objective(params, features, mask, targets):
    entropy = cross_entropy(params, features, mask, targets)
    penalty = L2 * (jnp.sum(params["w1"] ** 2) + jnp.sum(params["w2"] ** 2))
    return entropy + penalty, entropy


def zeros_like(params):
    return jax.tree.map(jnp.zeros_like, params)


def adam_step(params, moment, velocity, grads, step):
    moment = jax.tree.map(lambda m, g: BETA1 * m + (1.0 - BETA1) * g, moment, grads)
    velocity = jax.tree.map(
        lambda v, g: BETA2 * v + (1.0 - BETA2) * (g * g), velocity, grads
    )
    moment_hat = jax.tree.map(lambda m: m / (1.0 - BETA1**step), moment)
    velocity_hat = jax.tree.map(lambda v: v / (1.0 - BETA2**step), velocity)
    updated = jax.tree.map(
        lambda p, m, v: p - LEARNING_RATE * m / (jnp.sqrt(v) + ADAM_EPS),
        params,
        moment_hat,
        velocity_hat,
    )
    return updated, moment, velocity


@jax.jit
def train_batch(params, moment, velocity, features, mask, targets, step):
    grads, _entropy = jax.grad(objective, has_aux=True)(params, features, mask, targets)
    return adam_step(params, moment, velocity, grads, step)


def copy_params(params):
    return jax.tree.map(lambda array: jnp.array(array), params)


def to_list(array):
    return json.loads(json.dumps(array.tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    train_nodes, valid_nodes, train_ids, valid_ids = load_split(args.dataset)
    train_raw, train_mask, train_y, train_rows = pack(train_nodes)
    valid_raw, valid_mask, valid_y, _valid_rows = pack(valid_nodes)
    mu = jnp.mean(train_rows, axis=0)
    std = jnp.maximum(jnp.std(train_rows, axis=0), 1e-6)
    train_x = standardize(train_raw, mu, std)
    valid_x = standardize(valid_raw, mu, std)

    key = jax.random.PRNGKey(SEED)
    key, init_key = jax.random.split(key)
    params = init_params(init_key)
    moment = zeros_like(params)
    velocity = zeros_like(params)
    best_params = copy_params(params)
    best_validation = float("inf")
    best_epoch = 0
    train_history = []
    valid_history = []
    step = 0

    for epoch in range(1, EPOCHS + 1):
        key, shuffle_key = jax.random.split(key)
        order = jax.random.permutation(shuffle_key, train_x.shape[0])
        for start in range(0, train_x.shape[0], BATCH):
            batch = order[start : start + BATCH]
            step += 1
            params, moment, velocity = train_batch(
                params,
                moment,
                velocity,
                train_x[batch],
                train_mask[batch],
                train_y[batch],
                jnp.asarray(step, dtype=jnp.float64),
            )
        train_ce = float(cross_entropy(params, train_x, train_mask, train_y))
        valid_ce = float(cross_entropy(params, valid_x, valid_mask, valid_y))
        train_history.append(train_ce)
        valid_history.append(valid_ce)
        if epoch == 1 or epoch % 5 == 0:
            if valid_ce < best_validation:
                best_validation = valid_ce
                best_epoch = epoch
                best_params = copy_params(params)
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(
                f"epoch {epoch:3d}  train {train_ce:.6f}  val {valid_ce:.6f}",
                flush=True,
            )

    probe_params = best_params
    probes = []
    for node in valid_nodes[:6]:
        raw = jnp.asarray(node["features"], dtype=jnp.float64)
        standardized = standardize(raw, mu, std)
        probe_logits = logits_of(probe_params, standardized)
        log_probability = masked_log_softmax(probe_logits[None, :], jnp.ones((1, raw.shape[0]), dtype=bool))
        probability = jnp.exp(log_probability[0])
        probes.append(
            {
                "instance": int(node["instance"]),
                "candidates": [int(index) for index in node["candidates"]],
                "features": raw.tolist(),
                "logits": [float(value) for value in probe_logits],
                "probabilities": [float(value) for value in probability],
            }
        )

    document = {
        "w1": best_params["w1"].tolist(),
        "b1": best_params["b1"].tolist(),
        "w2": best_params["w2"].tolist(),
        "b2": float(best_params["b2"]),
        "mu": mu.tolist(),
        "std": std.tolist(),
        "history": {"train_ce": train_history, "val_ce": valid_history},
        "training": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "learning_rate": LEARNING_RATE,
            "l2": L2,
            "adam": {"beta1": BETA1, "beta2": BETA2, "eps": ADAM_EPS},
            "init": "glorot_normal",
            "std": "population over training candidate rows",
            "checkpoint_epoch": best_epoch,
            "checkpoint_val_ce": best_validation,
            "train_nodes": len(train_nodes),
            "validation_nodes": len(valid_nodes),
            "train_instances": sorted(train_ids),
            "validation_instances": sorted(valid_ids),
            "instance_overlap": 0,
        },
        "probes": probes,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(document, handle)
    print(
        f"checkpoint epoch {best_epoch} validation cross-entropy {best_validation:.6f}"
    )
    print(
        f"nodes train/val {len(train_nodes)}/{len(valid_nodes)} "
        f"instances {len(train_ids)}/{len(valid_ids)} overlap 0"
    )


if __name__ == "__main__":
    main()
