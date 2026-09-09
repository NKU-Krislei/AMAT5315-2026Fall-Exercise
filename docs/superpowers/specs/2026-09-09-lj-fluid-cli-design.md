# Lennard-Jones Fluid CLI (Part 4)

Date: 2026-09-09
Status: design sections approved; awaiting student review of this file
Crate: `week2/md`
Depends on: Part 3 dimer (`System`, `Integrator`, `VelocityVerlet`, `energy` / `force`)

## Goal

Turn the `md` crate into a CLI that runs an \(N=100\) Lennard-Jones fluid, writes a trajectory, checks energy drift / temperature / speed shape from the raw frames, and records a video of the atoms beside \(g(r)\). The Part 3 dimer (open boundaries, plain LJ, Euler vs Verlet) stays green.

## Non-goals

- No cell list (Part 5).
- No heating / melt (Part 5).
- Do not change dimer assertions or the `Integrator::step` signature.
- Do not put periodic wrap inside `Verlet::step`.
- Do not reimplement or finite-difference `energy(r)` / `force(r)`.
- Do not commit `artifacts/`, `target/`, PDFs, or MP4s except as VERIFY asks locally (the generated `fluid.mp4` stays untracked).

## Architecture (locked)

`System` carries an `Interaction`: `Open` (Part 3) or `Periodic { box_xy, rc }` (Part 4). `compute_accelerations` and `potential_energy` branch on it. Integrators are unchanged. The **run loop** wraps positions after each step when the interaction is periodic.

| Unit | Responsibility |
|------|----------------|
| `Interaction` on `System` | Open + full LJ, or PBC + \(r_c=2.5\) shifted LJ |
| `compute_accelerations` / `potential_energy` | All pairs; min-image and cutoff only if periodic |
| `Integrator` / `advance` | Same as Part 3 |
| Lattice + Maxwell-Boltzmann init | \(10\times 10\) triangular lattice; seeded Gaussians; remove COM; rescale |
| Run driver | Equilibrate with thermostat; produce without; write JSON |
| `md check` | Recompute physics from saved `pos`/`vel`; PASS/FAIL |
| `md video` | Atoms + \(g(r)\); `plotters` frames; `ffmpeg` MP4 |
| `week2/Makefile` | `make reproduce` → default release `run` into `week2/artifacts/` |

## Public API additions

```rust
#[derive(Clone, Debug)]
pub enum Interaction {
    Open,
    Periodic { box_xy: [f64; 2], rc: f64 },
}

impl System {
    pub fn new(positions: Vec<[f64; 2]>, velocities: Vec<[f64; 2]>) -> Self;
    // Open interaction, zero accelerations. Existing dimer constructor.

    pub fn periodic(
        positions: Vec<[f64; 2]>,
        velocities: Vec<[f64; 2]>,
        box_xy: [f64; 2],
        rc: f64,
    ) -> Self;

    pub fn n_atoms(&self) -> usize;
}

pub fn wrap_positions(system: &mut System); // no-op if Open
pub fn triangular_lattice(n: usize, rho: f64) -> (Vec<[f64; 2]>, [f64; 2]);
pub fn seed_maxwell_boltzmann(system: &mut System, temperature: f64, seed: u64);
pub fn rescale_temperature(system: &mut System, temperature: f64);

pub fn thermo_temperature(system: &System) -> f64; // 2 E_kin / (2N-2)
pub fn speed_temperature(velocities: &[[f64; 2]]) -> f64; // <v^2>/2

pub fn shifted_pair_energy(r: f64, rc: f64) -> f64; // U(r)-U(rc) if r<rc else 0
```

`System` fields stay public: `positions`, `velocities`, `accelerations`, plus `interaction: Interaction`. `System::new` sets `Interaction::Open` so the dimer test does not change.

`compute_accelerations` / `potential_energy` / `kinetic_energy` / `total_energy` keep their names. For `Periodic` they use minimum image and the shifted cutoff; for `Open` they behave exactly as Part 3.

## Forces and energy (periodic)

Minimum image, per axis: \(d \leftarrow d - L \cdot \mathrm{round}(d/L)\).

Cutoff \(r_c = 2.5\):

- \(r \ge r_c\): pair force 0, pair energy 0
- \(r < r_c\): pair force = existing `force(r)`; pair energy = `energy(r) - energy(rc)`

Vector force unchanged: \(\vec{d}=\vec{x}_i-\vec{x}_j\) after min-image, \(r=|\vec{d}|\), \(\vec{F}_i=F(r)\vec{d}/r\), \(\vec{F}_j=-\vec{F}_i\), \(m=1\).

Newton’s third law: \(\sum_i \vec{F}_i = 0\) within numerical noise.

After each `advance` in the **run loop only**, `wrap_positions` folds coordinates into \([0,L_x)\times[0,L_y)\) with `rem_euclid`. Velocities unchanged.

## Lattice

Density \(\rho=0.8\). \(a=\sqrt{2/(\sqrt{3}\rho)}\), \(h=\sqrt{3}/2\,a\). For \(N\) a square number: \(\sqrt{N}\) columns and rows, \(L_x=\sqrt{N}\,a\), \(L_y=\sqrt{N}\,h\).

Contract \(N=100\): \(i,j=0,\ldots,9\),

\[
x_{ij}=\bigl(i+\tfrac12(j \bmod 2)\bigr)a,\qquad y_{ij}=j h.
\]

Even row count so the stagger continues across the periodic \(y\) boundary.

## Init and thermostat

Draw each velocity component as independent Gaussian, mean 0, **variance** \(T\) (so std \(\sqrt{T}\)), RNG seed `seed` (contract `2026`). Subtract the mean velocity (remove COM). Then rescale every component by \(\sqrt{T_\mathrm{target}/T_\mathrm{thermo}}\) with

\[
T_\mathrm{thermo}=\frac{2E_\mathrm{kin}}{2N-2}.
\]

Rescale once after init, and again every 50 **equilibration** steps (including step 2000). Production: thermostat **off**. Production step counter starts at 0 after equilibration; no frame at production step 0.

Use `rand` `StdRng::seed_from_u64` plus Box–Muller (or equivalent). Pin a `rand` 0.8.x version so the seed is reproducible.

## Contract run (CLI defaults)

| Flag | Default |
|------|---------|
| `--n` | 100 |
| `--rho` | 0.8 |
| `--temperature` | 0.5 |
| `--dt` | 0.01 |
| `--eq-steps` | 2000 |
| `--steps` | 10000 |
| `--sample-every` | 50 |
| `--seed` | 2026 |
| `--out` | `artifacts` |
| `--integrator` | `velocity-verlet` |

`md run` with no extra flags equals that table. Integrator value in `run.json` is the string `"velocity-verlet"`.

Equilibration: 2000 Verlet steps, rescale every 50. Production: 10000 Verlet steps, no rescale, wrap every step. Save production steps with `step % sample_every == 0` and `step != 0` → **200 frames** (50, 100, …, 10000).

## Files in `--out`

**`run.json`:** `n`, `rho`, `box` as `[Lx, Ly]`, `dt`, `temperature`, `eq_steps`, `steps`, `sample_every`, `seed`, `integrator` as `"velocity-verlet"`.

**`traj.jsonl`:** one JSON object per saved production frame: `step`, `t` (`step * dt`), `pos` (wrapped), `vel`, `E_pot` (shifted potential), `E_kin`.

Create `--out` with `create_dir_all`. Use `serde` / `serde_json`.

## CLI (`clap` derive)

`src/main.rs` becomes:

```text
md run [flags]
md check <dir>
md video <dir> --out <mp4>
```

Library owns simulate / check / render. `greeting()` may remain for the existing unit test; `cargo run` no longer prints Hello World.

From `week2/`:

```bash
cargo run --manifest-path md/Cargo.toml --release -- run --out artifacts
cargo run --manifest-path md/Cargo.toml --release -- check artifacts
cargo run --manifest-path md/Cargo.toml --release -- video artifacts --out fluid.mp4
```

## `md check`

Recompute \(E_\mathrm{kin}\), \(E_\mathrm{pot}\) (shifted), speeds, and \(\chi^2\) **from saved positions and velocities**. Cross-check stored `E_pot` / `E_kin` against the recomputation (fail if they disagree beyond a tight tolerance, e.g. \(10^{-9}\) relative or absolute).

Let \(E_0\) be the recomputed total energy of the **first** saved frame. \(k=\max(1,\lfloor N_\mathrm{frames}/10\rfloor)\). Mean energy of the first \(k\) frames vs last \(k\) frames:

| Check | Pass |
|-------|------|
| Secular drift | \(\lvert\bar E_\mathrm{last\,k}-\bar E_\mathrm{first\,k}\rvert/|E_0| < 2\times 10^{-3}\) |
| Temperature | \(\lvert T_\mathrm{speed}-0.5\rvert < 0.05\), \(T_\mathrm{speed}=\langle v^2\rangle/2\) over **all** pooled production velocities |
| Speed shape | \(\chi^2_{22}<2\) |

Speed bins: pool \(v=|\vec{v}|\) from all saved frames. \(M\) speeds, 24 equal-probability bins at the **measured** \(T_\mathrm{speed}\):

\[
b_k=\sqrt{-2 T_\mathrm{speed}\ln(1-k/24)},\quad k=0,\ldots,23,\quad b_{24}=\infty.
\]

Expected count \(E_b=M/24\). \(O_b\) observed.

\[
\chi^2_{22}=\frac1{22}\sum_b\frac{(O_b-E_b)^2}{E_b}.
\]

Print each measured value beside its limit and `PASS` or `FAIL`. Exit non-zero if any check fails or files are malformed.

2D Maxwell–Boltzmann density (for the video/viewer, not a check threshold): \(f(v)=(v/T_\mathrm{speed})\exp(-v^2/(2 T_\mathrm{speed}))\).

## \(g(r)\) and video

\(g(r)\): min-image distances, rings out to half the **shorter** box side, divide by \(\rho\pi[(r+\Delta r)^2-r^2]\), average over atoms and frames.

`md video`: one output frame per saved trajectory frame; left panel atoms in the box; right panel \(g(r)\). Write temporary PNGs with `plotters`, encode with `ffmpeg` to `--out`. Fail clearly if `ffmpeg` is missing. File must be under 2 MB (ffmpeg flags such as a reasonable fps and quality). Do not commit the MP4.

## Makefile and gitignore

`week2/Makefile`:

```make
.PHONY: reproduce
reproduce:
	cargo run --manifest-path md/Cargo.toml --release -- run --out artifacts
```

`week2/.gitignore`: `artifacts/`, `*.mp4`. `md/.gitignore` already has `/target`.

## Tests (write first)

1. **`periodic_forces_sum_to_zero`** — several atoms in a periodic box with cutoff; \(\lvert\sum_i \vec a_i\rvert\) below a tight tolerance (e.g. \(10^{-12}\)).
2. **`cutoff_potential_continuous_from_inside`** — two atoms at \(r=r_c-10^{-4}\) (not only \(r=r_c\)); `potential_energy` equals `energy(r)-energy(rc)` to \(\sim 10^{-12}\); just outside \(r_c\), energy is 0. Unshifted `energy(r)` at that interior point is **not** \(\approx 0\), so forgetting the shift fails.
3. **`run_writes_run_json_and_traj_jsonl`** — invoke the built binary (or the library run entry) with **short** `eq_steps`/`steps`/`sample_every` but real flags; `run.json` has required fields; `traj.jsonl` has no step 0; frame count is `steps/sample_every`; positions wrapped. If `n<100`, keep \(r_c < L/2\) (square \(N\ge 36\) at \(\rho=0.8\)). \(N=100\) with a handful of steps is also fine.
4. **`contract_run_passes_physics`** — default contract through the library `run` + `check`. Skip the body under `debug_assertions` so `cargo test` stays fast; **`cargo test --release` must actually run it** and pass the three bounds. `make reproduce` + `md check` remains the instructor gate.

Existing `greeting` / `well_depth` / `force_matches_energy_derivative` / `dimer_euler_drifts_verlet_conserves` stay green.

## TDD / git order

1. Commit failing tests for (1) and (2) before production PBC code that would make them pass.
2. Implement `Interaction` + min-image + shifted energy/force until (1)(2) and dimer are green. Commit.
3. Commit failing (3) before the run writer.
4. Lattice, thermostat, run loop, JSON, `clap run`. Green (3). Commit.
5. `check` implementation; (4) as specified. Commit.
6. `Makefile` + gitignore. Commit.
7. `video` last, after check is green. Do not put video in a red-test commit.

No `git add .`. No push unless asked. Separate commits.

## Dependencies

- `clap` 4 with `derive`
- `serde`, `serde_json`
- `rand` 0.8 (`std_rng`)
- `plotters` already a **dev**-dependency; **move it to `[dependencies]`** (or add it there) so `md video` can use it from the binary
- `ffmpeg` on PATH at video time

## Success

- `cargo test --manifest-path week2/md/Cargo.toml --release` all green, including dimer and the four Part 4 checks (contract test actually executed in release).
- `make reproduce` in `week2/` writes `artifacts/run.json` and `artifacts/traj.jsonl`.
- `md check artifacts` prints three values within bounds and `PASS`.
- `md video` produces an MP4 < 2 MB with atoms that stay in the box and a liquid-like \(g(r)\).
