# Week 7: Diffusion and flow matching

Four equal Gaussian clusters, then a pinned CIFAR-10 noise predictor. Run the commands below from `week7/`. Figures are in `evidence/`. Checkpoints, raw arrays, downloaded weights, and `inputs/` stay local (see `.gitignore`). The locked environment is `uv.lock`.

```bash
uv sync
uv run pytest -q
```

Python 3.12.9, JAX 0.12.0, Optax 0.2.8, NumPy 2.5.4, Matplotlib 3.11.2, safetensors 0.8.0. Training and sampling use float32 on CPU.

## Training settings

`diffusion.toml` (seed 2026) and `flow.toml` (seed 2029) are the sheet's settings. Both use a three-layer width-128 SiLU MLP, 32 sinusoidal time features, Adam, and a cosine learning rate from 0.001 down to 0 over 25000 updates of batch 512. The loss is the mean squared error, which is `(1/d) ||pred - target||^2`. Diffusion features use level `t - 1`. Flow features use `1000 * s`.

The schedule has `T = 1000` and `beta_t` linear from `0.0001` to `0.02`. The four clusters are centred at `(3,0)`, `(0,3)`, `(-3,0)`, `(0,-3)`, with coordinate standard deviation `0.3`. Sampling does not clip.

## Evidence

| Figure | Command |
| --- | --- |
| `evidence/part1_forward.gif`, `evidence/part1_forward_frames.png` | `uv run python make_evidence.py forward` |
| `evidence/part1_ddpm_training.gif`, `evidence/part1_ddpm_training_frames.png` | `uv run python train_diffusion.py`, then `uv run python make_evidence.py ddpm-fit` |
| `evidence/part1_ddpm_samples.png`, `evidence/part1_ddpm_reverse.gif`, `evidence/part1_ddpm_metrics.txt` | `uv run python make_evidence.py ddpm-samples` |
| `evidence/part2_trajectories.png`, `evidence/part2_samples.png`, `evidence/part2_metrics.txt` | `uv run python make_evidence.py ddim` |
| `evidence/part3_velocity.png`, `evidence/part3_velocity.txt` | `uv run python train_flow.py`, then `uv run python make_evidence.py velocity` |
| `evidence/part3_steps.png`, `evidence/part3_steps.txt` | `uv run python make_evidence.py steps` |
| `evidence/part4_cifar_reverse.png`, `evidence/part4_cifar_reverse.gif`, `evidence/part4_cifar_compare.png` | `uv run python make_evidence.py cifar` |
| `evidence/part5_repaint.png`, `evidence/part5_repaint.txt` | `uv run python make_evidence.py repaint` |

The forward movie follows one batch of 2000 points. At `t = 1000` each coordinate is within 0.1 of mean 0 and variance 1. The trained DDPM sample (2000 points, 1000 steps) has cluster fractions 0.239 to 0.263, centres within 0.022 of the targets, and coordinate spreads 0.293 to 0.317.

Two 1000-step DDIM runs from the same noise differ by 0. On 2000 points the DDIM fractions stay within 0.015 of 0.25. One DDPM coordinate spread in that draw is 0.283, outside the reference band `[0.29, 0.32]` and inside 0.03 of 0.30. `part2_metrics.txt` records that the tight band was not met.

The learned velocity matches the exact marginal velocity on 4000 samples from the path. Relative RMS is 0.0095, 0.0187, and 0.0518 at `s = 0.1, 0.5, 0.9`.

## How many steps each sampler needs

`part3_steps.png` compares DDPM, DDIM, and flow matching on 4000 shared terminal samples. The shaded band on the energy plot is the 95% quantile of the energy distance between two independent exact samples of this size, 0.00105. The fraction band is two binomial standard errors at probability 0.25.

At 20 steps, flow matching has energy distance 0.00328 and width error 0.0273. DDIM is close (energy 0.00339). DDPM's width error is still 0.057 at 20 steps and falls to 0.024 at 50 steps. One step of DDPM and one step of DDIM are the same Tweedie estimate (energy 0.813). One Euler step of the flow collapses (energy 2.15) because the velocity has to be integrated, not applied once.

## Photographs

The CIFAR-10 predictor is `google/ddpm-cifar10-32` at revision `267b167dc01f0e4e61923ea244e8b988f84deb80`, loaded in JAX to match diffusers `v0.40.0` `UNet2DModel`. The sampler is the same 1000-step DDPM used on the mixture: variance `beta_t`, no clip, and the network receives mathematical level `t` as index `t - 1`. Colours are mapped with `(x + 1) / 2` only for display.

Seed 7041 draws four images. They stay noisy through `t = 200` and become a face, a dog, a textured animal, and an airplane by `t = 0`. `week6/README.md` only lists dataset URLs and has no gallery script, so the comparison row is the four real CIFAR-10 test images in `inputs/images.npz`.

## Inpainting

RePaint visits levels `1000, 980, ..., 20, 0`. Each jump except the last is tried twice, which is 99 network evaluations per completion. Between tries the merged state is noised back up. Observed pixels are replaced by the observation noised to the same level. Only the Tweedie clean estimate is clipped to `[-1, 1]`.

The mask hides rows and columns `6:26` in all three channels. Seeds `7201` to `7204` complete each of the four test images. Every observed pixel is unchanged (maximum change `0`). The 16 hidden-pixel MSEs average 0.204 and range from 0.089 to 0.302. The variance map is dark outside the square and bright inside it: the seeds agree on the visible pixels and disagree on the hole.

## Downloads

Weights and images are not committed.

```bash
mkdir -p inputs
curl -fL https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week7-images.npz \
  -o inputs/images.npz
uv run --with huggingface-hub==1.33.0 hf download google/ddpm-cifar10-32 \
  config.json diffusion_pytorch_model.safetensors \
  --revision 267b167dc01f0e4e61923ea244e8b988f84deb80 \
  --local-dir inputs/cifar-model
```
