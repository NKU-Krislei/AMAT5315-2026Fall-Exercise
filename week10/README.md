# Week 10: Approximate tensor contraction for the Ising model

Three hands-on builds from the learning sheet, all in this directory:

1. **SVD compression evidence** — a 512×512 Gaussian matrix, its spectrum, and the
   measured cost of truncated low-rank approximations (`svd` tool, `artifacts/svd.png`).
2. **Boundary-MPS contraction** — the open square-lattice Ising partition function by
   row absorption with an MPS boundary, peaks of the specific heat up to 64×64, and
   the extrapolated critical temperature (`thermo --method boundary`,
   `artifacts/boundary-critical.png`).
3. **TRG coarse-graining** — the same transition on periodic squares up to
   1024×1024 by tensor renormalization group (`thermo --method trg`,
   `artifacts/trg-critical.png`).

The model is the zero-field square-lattice Ising model with
`H(s) = -Σ_edges s_i s_j`, `s_i ∈ {+1, −1}`, `J = k_B = 1`, at inverse temperature
`K = 1/T`. Every run states its geometry (open or periodic) explicitly; the tools
take no defaults (`thermo.design.toml` is the interface contract).

## Environment

| component | version |
|---|---|
| rustc / cargo | 1.99.1 (stable toolchain via `rust-toolchain.toml`) |
| faer | 0.24.4 (SVD; `dyn-stack` 0.13.2 for its workspace) |
| rand / rand_distr | 0.9.5 / 0.5.1 (Part 1 generator, see below) |
| serde / serde_json | 1.0.229 / 1.0.154 (evidence files) |
| Python (plots, cross-checks) | 3.12 with numpy 1.26.4, matplotlib 3.11.2 |

Random generator record (Part 1): entries are drawn with
`rand_distr::StandardNormal` (ziggurat) from a `rand::rngs::StdRng` (ChaCha12 core)
seeded with `SeedableRng::seed_from_u64(2026)`; crate versions are pinned in
`Cargo.lock`. The generator is also recorded in `runs/svd/meta.json`.

Note on faer: 0.20.2 returns inconsistent singular *vectors* for rank-deficient
inputs (the Ising tensors have many exactly-zero singular values), which broke
χ-capped truncation. 0.24.4 is correct; the pin is deliberate.

## Layout

```
src/svd.rs          SVD wrapper (row-major, sorted) + Algorithm 1 (TruncatedSVD)
                    + Algorithm 2 (Split, with full spectrum diagnostics)
src/ising.rs        W, site tensor A, open-edge vector b, spin enumeration
src/boundary.rs     Algorithm 3 (canonicalizing compression) + Algorithm 4
src/trg.rs          Algorithm 5 (normalize, split, regroup, close)
src/svd_evidence.rs Part 1 pipeline (generate / independently check)
src/main.rs         the thermo CLI; src/svd_bin.rs the svd CLI
tests/              all acceptance checks from the sheet (see Verification)
sweep_boundary.py   Part 2 sweep driver; sweep_trg.py Part 3
peaks.py            shared c_V / parabola / fit helpers
plot_svd.py, plot_boundary.py, plot_trg.py   figures (write artifacts/)
recheck.py          independent recomputation of log Z from saved evidence
thermo.design.toml  the tool's interface description
runs/ artifacts/    per-run evidence and figures (kept out of Git)
```

## Part 1: SVD compression evidence

```bash
cargo run --release --quiet --bin svd -- gen --out runs/svd
cargo run --release --quiet --bin svd -- check --out runs/svd
python3 plot_svd.py runs/svd          # numpy cross-check + artifacts/svd.png
```

`check` recomputes the relative Frobenius errors from the saved matrices only
(Rust), and `plot_svd.py` repeats the computation independently with numpy:

| χ | e measured | e predicted | \|diff\| | factor storage / original |
|---|---|---|---|---|
| 16 | 0.941707835 | 0.941707835 | 1.1e-15 | 0.062 |
| 64 | 0.788668898 | 0.788668898 | 4.4e-16 | 0.250 |
| 128 | 0.612646556 | 0.612646556 | 5.6e-16 | 0.500 |
| 256 | 0.323841710 | 0.323841710 | 5.6e-17 | 1.000 |
| 512 | 8.5e-15 | 0 | 8.5e-15 | 2.000 |

Every measured error agrees with the discarded-spectrum prediction
`sqrt(Σ_{α>χ} s_α²)/‖M‖_F` within 8.5e-15 (requirement 1e-8), and the full-rank
error is 8.5e-15 (requirement 1e-8). Two-factor storage equals the original at
χ = 256; the error first reaches 10% only at χ = 394, i.e. 1.539× the original
storage. **The Gaussian matrix admits no storage-saving low-rank approximation
at 10% error** — the spectrum is far too flat (a broad-spectrum warning for the
lattice methods that follow).

## Part 2: boundary MPS on open squares

```bash
cargo build --release
cargo install --locked --path .          # optional; or use target/release/thermo
thermo --method boundary --l 64 --k 0.45 --chi 16 --out runs/boundary64
python3 sweep_boundary.py               # broad + refined runs (~180 runs)
python3 plot_boundary.py                # fits + artifacts/boundary-critical.png
python3 recheck.py runs/boundary64      # recompute log Z from saved cores
```

The stated case gives `g = log Z / L² = 0.936885160478` (0.8 s). The sweep covers
L = 16, 32, 64 over K = 0.40…0.50 step 0.0025 at χ = 16, then nine points spaced
0.00125 around each broad peak at χ = 16 and χ = 32; peaks are fitted with a
three-point parabola in K.

| L | K_peak | T_peak | c_V at peak |
|---|---|---|---|
| 16 | 0.47776 | 2.0931 | 1.115 |
| 32 | 0.45878 | 2.1797 | 1.388 |
| 64 | 0.44960 | 2.2242 | 1.683 |

Fitting `T_peak(L) = T_c + a/L` (χ = 32, h = 0.00125): **T_c = 2.2675** using all
three sizes and **2.2688** using the largest two — 0.074% and 0.019% from Onsager's
`2/log(1+√2) = 2.269185`. Peak positions agree within 0.001% under the rank change
(χ = 16 → 32) and the step change (h = 0.0025 → 0.00125). The peaks grow and move
toward the transition as the magnet grows.

## Part 3: TRG on periodic squares

```bash
thermo --method trg --l 1024 --k 0.44 --chi 16 --out runs/trg1024   # 0.4 s
python3 sweep_trg.py                  # L = 64, 256, 1024, same protocol
python3 plot_trg.py                   # fits + artifacts/trg-critical.png
python3 recheck.py runs/trg1024
```

| L | K_peak | T_peak | c_V at peak |
|---|---|---|---|
| 64 | 0.43829 | 2.2816 | 2.249 |
| 256 | 0.44009 | 2.2723 | 2.903 |
| 1024 | 0.44053 | 2.2700 | 3.096 |

**T_c = 2.2692** using all three sizes and **2.2693** using the largest two —
0.0009% and 0.0034% from Onsager. Periodic peaks approach the transition from
above (open peaks approached from below) and peak positions are stable to
< 0.1% under the rank and step changes. Solver time: ~0.4 s per 1024×1024 run at
χ = 16, ~23 s at χ = 32 (single run, 4 threads).

## Verification

`cargo test --release` runs all acceptance checks from the sheet:

| check | tolerance | measured |
|---|---|---|
| open 4×4, χ=8, g vs enumeration at K=0, 0.44, 0.7 | < 1e-8 | 0, 1.3e-14, 4.4e-14 |
| open 4×4, c_V (h=0.0025) vs K²·Var(H)/16 at K=0.44 | < 1e-3 | 3.8e-6 (0.439393 vs 0.439397) |
| periodic L=2, χ=16, g vs enumeration (double bonds counted) | < 1e-8 | ≤ 4.4e-16 |
| periodic L=4, χ=8, g vs enumeration at K=0.44 | < 2e-3 | 1.1e-4 |
| periodic L=2, c_V (h=0.0025) vs K²·Var(H)/4 | < 1e-3 | 3.7e-6 (0.404282 vs 0.404278) |
| both geometries, g at K=0 | = log 2 | exact to 1e-10 |
| SVD truncated error vs discarded spectrum | < 1e-10 | 1e-15 (unit test) |

`recheck.py` recomputes `log Z` from the saved final cores / final tensor and the
recorded scales alone; it reproduces the stored values to 0.0e+00 (exact bit
round-trip through the saved evidence).

## Reproduction

```bash
cd week10
cargo build --release
cargo test --release                                   # verification table above
cargo run --release --quiet --bin svd -- gen --out runs/svd
cargo run --release --quiet --bin svd -- check --out runs/svd
python3 plot_svd.py
./target/release/thermo --method boundary --l 64 --k 0.45 --chi 16 --out runs/boundary64
./target/release/thermo --method trg --l 1024 --k 0.44 --chi 16 --out runs/trg1024
python3 sweep_boundary.py && python3 plot_boundary.py
python3 sweep_trg.py && python3 plot_trg.py
python3 recheck.py runs/boundary64 runs/trg1024
```

Each `thermo` run writes `run.json` (model, geometry, settings, log Z, g, solver
time, scales, truncation diagnostics) plus binary tensors (`final_cores.bin` /
per-step `stepNN_factors.bin` + `final_tensor.bin`) at full f64 precision — enough
to recompute the final contraction independently. `runs/` and `artifacts/`
(g(K) tables, peak-stencil runs, figures) are generated data and are kept out of
Git, per the sheet; regenerate them with the commands above.
