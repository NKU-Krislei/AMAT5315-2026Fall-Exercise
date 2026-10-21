"""CIFAR-10 DDPM U-Net, matching diffusers v0.40.0 UNet2DModel.

The checkpoint is google/ddpm-cifar10-32 at the revision pinned by the sheet.
Attention weights still use the deprecated query/key/value/proj_attn names.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open

from schedule import NoiseSchedule

ROOT = Path(__file__).resolve().parent
CIFAR_WEIGHTS = ROOT / "inputs" / "cifar-model" / "diffusion_pytorch_model.safetensors"
GROUPS = 32
NORM_EPS = 1e-6
ATTENTION_SCALE = 256.0**-0.5


def load_cifar_weights(path=CIFAR_WEIGHTS):
    weights = {}
    with safe_open(path, framework="numpy") as handle:
        for key in handle.keys():
            weights[key] = jnp.asarray(handle.get_tensor(key))
    return weights


def timestep_embedding(timesteps, dim=128, max_period=10000.0, freq_shift=1.0):
    """Sinusoidal embedding with flip_sin_to_cos false and freq_shift 1."""
    half = dim // 2
    exponent = -jnp.log(jnp.asarray(max_period, dtype=jnp.float32)) * jnp.arange(half, dtype=jnp.float32)
    exponent = exponent / (half - freq_shift)
    angles = timesteps.astype(jnp.float32)[:, None] * jnp.exp(exponent)[None, :]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


def _linear(x, weight, bias):
    return x @ jnp.swapaxes(weight, -1, -2) + bias


def _conv(x, weight, bias, stride=1, padding=0):
    kernel = jnp.transpose(weight, (2, 3, 1, 0))
    channels_last = jnp.transpose(x, (0, 2, 3, 1))
    output = jax.lax.conv_general_dilated(
        channels_last,
        kernel,
        window_strides=(stride, stride),
        padding=[(padding, padding), (padding, padding)],
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    output = jnp.transpose(output, (0, 3, 1, 2))
    return output + bias.reshape(1, -1, 1, 1)


def _group_norm(x, weight, bias, groups=GROUPS, eps=NORM_EPS):
    batch, channels, height, width = x.shape
    grouped = x.reshape(batch, groups, channels // groups, height, width)
    mean = jnp.mean(grouped, axis=(2, 3, 4), keepdims=True)
    variance = jnp.var(grouped, axis=(2, 3, 4), keepdims=True)
    normalized = (grouped - mean) * jnp.reciprocal(jnp.sqrt(variance + eps))
    normalized = normalized.reshape(batch, channels, height, width)
    return normalized * weight.reshape(1, -1, 1, 1) + bias.reshape(1, -1, 1, 1)


def _silu(x):
    return jax.nn.silu(x)


def _resnet(x, temb, weights, prefix):
    hidden = _silu(_group_norm(x, weights[f"{prefix}.norm1.weight"], weights[f"{prefix}.norm1.bias"]))
    hidden = _conv(hidden, weights[f"{prefix}.conv1.weight"], weights[f"{prefix}.conv1.bias"], padding=1)
    time_shift = _linear(
        _silu(temb), weights[f"{prefix}.time_emb_proj.weight"], weights[f"{prefix}.time_emb_proj.bias"]
    )
    hidden = hidden + time_shift[:, :, None, None]
    hidden = _silu(_group_norm(hidden, weights[f"{prefix}.norm2.weight"], weights[f"{prefix}.norm2.bias"]))
    hidden = _conv(hidden, weights[f"{prefix}.conv2.weight"], weights[f"{prefix}.conv2.bias"], padding=1)
    shortcut = f"{prefix}.conv_shortcut.weight"
    if shortcut in weights:
        x = _conv(x, weights[shortcut], weights[f"{prefix}.conv_shortcut.bias"])
    return x + hidden


def _attention(x, weights, prefix):
    residual = x
    batch, channels, height, width = x.shape
    hidden = _group_norm(x, weights[f"{prefix}.group_norm.weight"], weights[f"{prefix}.group_norm.bias"])
    sequence = jnp.transpose(hidden.reshape(batch, channels, height * width), (0, 2, 1))
    query = _linear(sequence, weights[f"{prefix}.query.weight"], weights[f"{prefix}.query.bias"])
    key = _linear(sequence, weights[f"{prefix}.key.weight"], weights[f"{prefix}.key.bias"])
    value = _linear(sequence, weights[f"{prefix}.value.weight"], weights[f"{prefix}.value.bias"])
    scores = jnp.einsum("bqc,bkc->bqk", query, key) * ATTENTION_SCALE
    probabilities = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
    mixed = jnp.einsum("bqk,bkc->bqc", probabilities, value)
    projected = _linear(mixed, weights[f"{prefix}.proj_attn.weight"], weights[f"{prefix}.proj_attn.bias"])
    restored = jnp.transpose(projected, (0, 2, 1)).reshape(batch, channels, height, width)
    return restored + residual


def _downsample(x, weights, prefix):
    padded = jnp.pad(x, ((0, 0), (0, 0), (0, 1), (0, 1)))
    return _conv(padded, weights[f"{prefix}.weight"], weights[f"{prefix}.bias"], stride=2)


def _upsample(x, weights, prefix):
    enlarged = jnp.repeat(jnp.repeat(x, 2, axis=2), 2, axis=3)
    return _conv(enlarged, weights[f"{prefix}.weight"], weights[f"{prefix}.bias"], padding=1)


def _time_embedding(timesteps, weights):
    embedded = timestep_embedding(timesteps)
    hidden = _silu(_linear(embedded, weights["time_embedding.linear_1.weight"], weights["time_embedding.linear_1.bias"]))
    return _linear(hidden, weights["time_embedding.linear_2.weight"], weights["time_embedding.linear_2.bias"])


def _down_block(hidden, temb, skips, weights, prefix, layers, attend, downsample):
    for index in range(layers):
        hidden = _resnet(hidden, temb, weights, f"{prefix}.resnets.{index}")
        if attend:
            hidden = _attention(hidden, weights, f"{prefix}.attentions.{index}")
        skips.append(hidden)
    if downsample:
        hidden = _downsample(hidden, weights, f"{prefix}.downsamplers.0.conv")
        skips.append(hidden)
    return hidden, skips


def _up_block(hidden, temb, skips, weights, prefix, layers, attend, upsample):
    for index in range(layers):
        hidden = jnp.concatenate([hidden, skips.pop()], axis=1)
        hidden = _resnet(hidden, temb, weights, f"{prefix}.resnets.{index}")
        if attend:
            hidden = _attention(hidden, weights, f"{prefix}.attentions.{index}")
    if upsample:
        hidden = _upsample(hidden, weights, f"{prefix}.upsamplers.0.conv")
    return hidden


def cifar_unet(images, timesteps, weights):
    """Predict noise. ``timesteps`` are checkpoint indices, shape (batch,)."""
    temb = _time_embedding(timesteps, weights)
    hidden = _conv(images, weights["conv_in.weight"], weights["conv_in.bias"], padding=1)
    skips = [hidden]
    hidden, skips = _down_block(hidden, temb, skips, weights, "down_blocks.0", 2, False, True)
    hidden, skips = _down_block(hidden, temb, skips, weights, "down_blocks.1", 2, True, True)
    hidden, skips = _down_block(hidden, temb, skips, weights, "down_blocks.2", 2, False, True)
    hidden, skips = _down_block(hidden, temb, skips, weights, "down_blocks.3", 2, False, False)
    hidden = _resnet(hidden, temb, weights, "mid_block.resnets.0")
    hidden = _attention(hidden, weights, "mid_block.attentions.0")
    hidden = _resnet(hidden, temb, weights, "mid_block.resnets.1")
    hidden = _up_block(hidden, temb, skips, weights, "up_blocks.0", 3, False, True)
    hidden = _up_block(hidden, temb, skips, weights, "up_blocks.1", 3, False, True)
    hidden = _up_block(hidden, temb, skips, weights, "up_blocks.2", 3, True, True)
    hidden = _up_block(hidden, temb, skips, weights, "up_blocks.3", 3, False, False)
    hidden = _silu(_group_norm(hidden, weights["conv_norm_out.weight"], weights["conv_norm_out.bias"]))
    return _conv(hidden, weights["conv_out.weight"], weights["conv_out.bias"], padding=1)


def sample_cifar_ddpm(weights, schedule: NoiseSchedule, key, n_images=4, record_levels=(1000, 500, 200, 100, 50, 0)):
    """1000-step DDPM. The network receives mathematical level t as index t - 1."""

    @jax.jit
    def step(images, noise, level, beta, alpha, alpha_bar):
        prediction = cifar_unet(images, jnp.full((images.shape[0],), level - 1), weights)
        mean = (1.0 / jnp.sqrt(alpha)) * (images - (beta / jnp.sqrt(1.0 - alpha_bar)) * prediction)
        fresh = jnp.where(level > 1, noise, jnp.zeros_like(images))
        return mean + jnp.sqrt(beta) * fresh

    key, start_key = jax.random.split(key)
    images = jax.random.normal(start_key, (n_images, 3, 32, 32), dtype=jnp.float32)
    frames = {}
    record = set(record_levels)
    if 1000 in record:
        frames[1000] = np.asarray(images)
    for level in range(int(schedule.betas.shape[0]), 0, -1):
        key, noise_key = jax.random.split(key)
        noise = jax.random.normal(noise_key, images.shape, dtype=jnp.float32)
        images = step(
            images,
            noise,
            jnp.int32(level),
            schedule.betas[level - 1],
            schedule.alphas[level - 1],
            schedule.alpha_bar[level],
        )
        if (level - 1) in record:
            frames[level - 1] = np.asarray(images)
        if level % 100 == 0 or level == 1:
            print(f"CIFAR DDPM level {level - 1}", flush=True)
    return np.asarray(images), frames


def to_display(images):
    """Map clean colours from [-1, 1] to [0, 1] for viewing only."""
    shown = (np.asarray(images) + 1.0) / 2.0
    return np.clip(shown, 0.0, 1.0)
