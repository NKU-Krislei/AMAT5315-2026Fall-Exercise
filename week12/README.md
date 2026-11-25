# Week 12: Integer programming and learned branching

Spin glasses are solved as mixed-integer linear programs. The LP relaxation gives a lower bound, branch-and-bound recovers the exact ground state, and a small network imitates Strong Branching without extra LP solves.

Run every command from `week12/`. Generated runs and figures stay in `artifacts/` and are not committed.

## Environment

- Rust 1.98.1, Cargo, CMake 4.0.2
- `pred` 0.6.0 (`cargo install problemreductions-cli@0.6.0 --locked`)
- `highs` crate 2.4.0, presolve off, one thread, LP solved by simplex
- `uv` 0.13.4 with JAX 0.12.1 in 64-bit mode on CPU, and matplotlib for the figures

`pred` 0.6.0 prints a panicked-thread warning before the answer. The warning is harmless.

## Settings

- Seeds: `2026` for the gap and branch-and-bound benchmarks, `3026` for the 16 unseen 22-spin graphs, `4026` for the eight 30-spin graphs.
- The per-instance SplitMix64 state is the first `next()` return value of a generator whose state is `seed XOR (id * 0xD1B54A32D192ED03)`.
- Branch-and-bound is depth-first. A node is pruned when its LP is infeasible or its bound is at least the incumbent minus `1e-6`. Ties break toward the lowest spin index. At `x = 0.5`, the `x = 1` child is visited first.
- The network is a shared MLP, 12 inputs, 48 `tanh` units, one score. `W1` is 12 by 48. Glorot normal initialization and the Adam shuffle use seed `2026`.
- Adam uses learning rate `4e-3`, batch size 64, 120 epochs, `beta1 = 0.9`, `beta2 = 0.999`, and `eps = 1e-8`.
- The loss is masked cross-entropy plus `2e-4` times the sum of squared weights. Biases are not penalized. Feature statistics are the population mean and standard deviation over training candidate rows, with standard deviation floored at `1e-6`.
- The checkpoint is the lowest validation cross-entropy among epoch 1 and every fifth epoch. This run kept epoch 95, validation cross-entropy 1.738673.

## Reproduce

```bash
mkdir -p artifacts/pred
cd artifacts/pred
pred create SpinGlass --graph 0-1,1-2,2-3,3-0 --couplings=1,1,1,-1 -o square.json
pred solve square.json
pred create SpinGlass --random --num-vertices 100 --edge-prob 0.04 --seed 2026 -o r100.json
time pred solve r100.json
cd ../..

cargo install --path . --force
branchopt --mode gap --spins 22 --edges 52 --count 16 --seed 2026 --out artifacts/gap
branchopt --mode bnb --spins 22 --edges 52 --count 48 --policy most-frac \
  --seed 2026 --out artifacts/bnb_most
branchopt --mode bnb --spins 22 --edges 52 --count 48 --policy strong \
  --export-labels --seed 2026 --out artifacts/bnb_strong

uv run --with jax==0.12.1 python train_branch.py \
  --dataset artifacts/bnb_strong/dataset.json --out artifacts/model.json

branchopt --mode eval --spins 22 --edges 52 --count 16 --seed 3026 \
  --model artifacts/model.json --out artifacts/eval_test
branchopt --mode eval --spins 30 --edges 72 --count 8 --seed 4026 \
  --model artifacts/model.json --out artifacts/eval_transfer

uv run --with matplotlib python scripts/plot_gap.py
uv run --with matplotlib python scripts/plot_bnb.py
uv run --with matplotlib python scripts/plot_neural.py
uv run --with matplotlib python scripts/plot_eval.py
```

The strong branch-and-bound command above also writes `artifacts/bnb_strong/dataset.json`. Node counts do not depend on that export.

## Checks from this machine

- Square: LP `-4` at `(0.5, 0.5, 0.5, 0.5)`, exact `-2`, nearest rounding `+2`. Diet cost `23.071429`.
- Sixteen 22-spin graphs: every LP bound is `-52`, mean exact energy `-38.25`, mean gap `13.75`, all 22 spins fractional, rounding misses every ground state, mean excess `34.25` (60 on instance 11).
- `pred` reports `Min(-2)` and `Min(-138)`.
- Frustrated tetrahedron: five nodes, bounds `-6`, `-4`, incumbent `-3`, prune at `-3`, prune at `-2`.
- Forty-eight graphs: most-frac mean 50.875 nodes and LP solves; Strong Branching mean 16 nodes and 287.417 LP solves. Both match HiGHS. Strong Branching is about 5.7 times slower on this laptop (61.59 ms versus 10.80 ms).
- Labels: 276 training nodes from instances 0–35 and 84 validation nodes from instances 36–47, no shared instance.
- Rust and JAX probe probabilities differ by at most `5.6e-16`. Every searched energy matches HiGHS within `4e-13`. For most-frac, neural, and root-neural, LP solves equal nodes.
- Sixteen unseen 22-spin graphs: mean nodes 50.875 (most-frac), 15.250 (neural), 15.625 (strong), 15.500 (root-neural). The largest neural saving is instance 4, energy `-38`: most-frac splits spins 0 then 1 and uses 157 nodes; neural splits spins 4 then 21 and uses 19 nodes.
- Eight 30-spin graphs, without retraining: mean nodes 177.75, 38, 40, and 51.5 for most-frac, strong, neural, and root-neural. Mean search times are 54.4 ms, 278.0 ms, 12.9 ms, and 15.5 ms. Feature-plus-inference time is 0.32 ms for neural and 0.02 ms for root-neural.
