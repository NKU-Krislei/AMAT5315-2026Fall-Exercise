# Week 2 code review

Review of `week2/` against `docs/superpowers/specs/2026-09-16-cell-list-heating-design.md` and `docs/superpowers/plans/2026-09-16-cell-list-heating.md`, plus the Part 3–4 dimer and fluid designs.

Date: 2026-09-16

## Summary

Cell-list forces default on `md run`, naive remains behind `--force naive`, and `--ramp-to` heats production. Release tests, including the \(N=100\) contract physics check and the four naive-vs-cells equality cases, pass.

## Findings

### Force paths agree on the sheet cases — fixed

`week2/md/tests/cells.rs` compares accelerations and potential energy for a perturbed lattice, a pair across the periodic boundary, pairs just inside and outside \(r_c\), and a two-cell-wide box (\(L=5\), \(r_c=2.5\)). Neighbour-cell indices are wrapped and deduplicated.

Commit: `8b36cae`

### Heating schedule is tested — fixed

`ramp_target(0.2, 1.2, 0, N)=0.2`, last step \(=1.2\), midpoint \(=0.7\). `md run --ramp-to 1.2` writes `ramp_to` in `run.json` and the last sampled thermo temperature matches the schedule after the production rescale.

Commit: `daffd73`

### Integrators unchanged — fixed

`Integrator::step` is still `fn step(&self, system: &mut System, dt: f64)`. Periodic wrap stays in the run loop, not in Verlet. Open-boundary dimer tests remain green.

Commit: `8b36cae` (no integrator edits)

### `md check` still assumes \(T=0.5\) — not fixed

`check.rs` passes only if speed temperature is within \(0.05\) of \(0.5\). A heating trajectory is supposed to leave that window. Keep heating `--out` away from `artifacts/` and do not run `md check` on `docs/`.

Reason: Part 4 contract, not a heating validator. Changing it would break the unheated PASS table.

### Cell-list occupancy vectors are cloned every evaluation — not fixed

`for_each_unique_pair` clones each cell's index list before visiting neighbours. That is extra allocation, visible as `build_cells` / drop glue in the cell-list profile, but N=1600 still speedups \(>2\).

Reason: clarity over micro-optimisation; equality tests already lock the pair set.

### At \(N=100\) cells are slower than naive — not fixed

Benchmark median speedup at \(N=100\) is \(0.78\). Building the grid costs more than the tiny all-pairs loop. The README states this; the speedup column still rises with \(N\) and exceeds 2 at \(N=1600\).

Reason: the table is the measurement. Claiming a \(N=100\) win would contradict it.

## Recheck after this review

```bash
cargo test --manifest-path week2/md/Cargo.toml --release
```

Expected: all unit, cells, dimer, pbc, and CLI tests pass, including `contract_run_passes_physics`.
