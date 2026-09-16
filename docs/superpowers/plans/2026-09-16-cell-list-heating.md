# Cell List and Heating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a cell-list periodic force path behind `--force`, default `cells`, that matches naive forces and energy, plus `--ramp-to` linear production heating.

**Architecture:** `ForceMode` on `System` and `RunParams`. Periodic `compute_accelerations` / `potential_energy` rebuild a wrapped, deduplicated 9-cell neighbour list or fall back to all pairs. Integrators unchanged. Production thermostat uses `ramp_target`.

**Tech Stack:** Rust `md` crate, clap 4, serde_json, existing Verlet + shifted LJ.

## Global Constraints

- Reuse `energy(r)` / `force(r)`; do not reimplement or finite-difference the force.
- `Integrator::step(&self, system: &mut System, dt: f64)` unchanged. Do not wrap inside Verlet.
- `System::new` remains Open-boundary. Open always uses all-pairs.
- Cell rule: \(n_x=\lfloor L_x/r_c\rfloor\), width \(L_x/n_x\); nine wrapped, deduplicated neighbour cells; rebuild every evaluation.
- `--force cells` default; `--force naive` keeps the original loop.
- `--ramp-to` is optional; \(T(s)=T_0+(T_1-T_0)s/N\); rescale every 50 production steps; `run.json` records `ramp_to`.
- Dimer / pbc / CLI tests stay green. No PDFs, `artifacts/`, `target/`, `week2-sim.py`.

## File structure

- Modify: `week2/md/src/system.rs` — `ForceMode`, cell list, branch in force/energy
- Modify: `week2/md/src/simulate.rs` — `RunParams.force`, `ramp_to`, production thermostat
- Modify: `week2/md/src/lib.rs` — export `ForceMode`, `ramp_target`
- Modify: `week2/md/src/main.rs` — `--force`, `--ramp-to`
- Create: `week2/md/tests/cells.rs` — equality cases
- Modify: `week2/md/tests/cli.rs` — heating schedule / `ramp_to` in `run.json`

---

### Task 1: Failing cell-list equality tests

**Files:**
- Create: `week2/md/tests/cells.rs`

**Interfaces:**
- Consumes: wished-for `ForceMode`, `System.force_mode`, `compute_accelerations`, `potential_energy`
- Produces: four failing tests (perturbed lattice, wrap pair, cutoff pair, two-cell box)

- [ ] **Step 1: Write the tests**

Create `week2/md/tests/cells.rs` comparing naive vs cells on the four sheet cases. Helper copies a system, sets `force_mode`, and asserts accelerations and energy within `1e-12`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n modernSC cargo test --manifest-path week2/md/Cargo.toml --test cells`

Expected: FAIL because `ForceMode` does not exist.

- [ ] **Step 3: Add `ForceMode` and wire naive/cells; implement cell list**

Minimal: enum on `System`, default `Cells` for `periodic`, `Naive` ignored on Open. Shared pair application. Cell bins as specified; fallback to naive if \(n_x<1\) or \(n_y<1\).

- [ ] **Step 4: Re-run cells tests**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add week2/md/tests/cells.rs week2/md/src/system.rs week2/md/src/lib.rs
git commit -m "Add a cell-list force path that matches naive pairs."
```

---

### Task 2: CLI `--force` default cells

**Files:**
- Modify: `week2/md/src/simulate.rs`, `week2/md/src/main.rs`, `week2/md/src/lib.rs`

**Interfaces:**
- Consumes: `ForceMode`
- Produces: `RunParams.force`; `md run --force naive|cells`

- [ ] **Step 1: Set `system.force_mode` from `RunParams` in `run_simulation`**
- [ ] **Step 2: Add clap `--force` with default `cells`**
- [ ] **Step 3: `cargo test --manifest-path week2/md/Cargo.toml` — dimer/pbc/cli still pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "Default md run to cell-list forces, keep --force naive."
```

---

### Task 3: Heating schedule

**Files:**
- Modify: `week2/md/src/simulate.rs`, `week2/md/src/main.rs`, `week2/md/tests/cli.rs`, `week2/md/src/lib.rs`

**Interfaces:**
- Produces: `pub fn ramp_target(t0: f64, t1: f64, step: usize, n_steps: usize) -> f64`
- `RunParams.ramp_to: Option<f64>`
- `run.json` field `ramp_to`

- [ ] **Step 1: Failing unit test** for endpoints (`step=0` → `t0`, `step=n_steps` → `t1`) and midpoint, plus CLI test that `--ramp-to 1.2` is stored.
- [ ] **Step 2: Watch fail**
- [ ] **Step 3: Implement `ramp_target`, production rescale every 50 steps, clap flag**
- [ ] **Step 4: Tests pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "Raise the production thermostat linearly with --ramp-to."
```

---

### Task 4: Measure, publish, review (after code is green)

Profile cells, benchmark N=100/400/1600, `scaling.png`, GitHub Pages viewer, heating run into `docs/`, `cold.mp4` / `hot.mp4`, `REVIEW.md`, README tables, push, Issue `@GiggleLiu` `@isPANN`.
