# Two-Atom Molecular Dynamics (Part 3 Dimer)

Date: 2026-09-09
Status: approved architecture + student deltas; awaiting student review of this file
Crate: `week2/md`

## Goal

Extend the existing `md` crate so two atoms move under plain Lennard-Jones forces with either forward Euler or velocity-Verlet, selected through one `Integrator` trait. The experiment’s output is the relative total-energy error time series. A later example plots that series; it is not part of the red test commit.

## Non-goals

- Do not change `src/main.rs` (it stays Hello World).
- Do not reimplement `energy(r)` / `force(r)`, and do not obtain the force by finite-differencing the energy.
- No cutoff, no periodic boundaries, no thermostat, no `FreeFlight`, no time-reversal test.
- Verlet’s 5000-step run is for `examples/dimer.rs` only, not for the unit test.
- Do not write `examples/dimer.rs` in the same commit as the failing dimer test.

## Architecture (locked)

Library owns physics and integration. Executable stays Hello World. Plotting follows `examples/field.rs`.

| Unit | Responsibility | Depends on |
|------|----------------|------------|
| `energy` / `force` | Scalar pair \(U(r)\), \(F(r)\) (already in `src/lib.rs`) | none |
| `System` | Owns `positions`, `velocities`, **and** `accelerations` | none |
| `compute_accelerations` | Open-boundary all-pairs \(\vec{F}\), writes `accelerations` | `force`, `System` |
| `Integrator` + `advance` | `step(&self, &mut System, dt)`; one driver | `System` |
| `Euler` / `VelocityVerlet` | Two update rules | `compute_accelerations` |
| `relative_energy_errors` | Run `n_steps`, return \((E(t)-E_0)/\|E_0\|\) after every step | `advance`, energy helpers |
| dimer test | Same initial state; only the value passed to `advance` differs | `relative_energy_errors` |
| `examples/dimer.rs` | Plot errors → `week2/dimer.png` | `relative_energy_errors` |

Data flow: build dimer state → `compute_accelerations` from the initial positions → loop `advance` → after each step record relative energy error. Euler and Verlet differ only by which `&impl Integrator` is passed.

Part 1’s reading example omitted accelerations. Storing them on `System` is an intentional extension so Verlet can reuse \(\vec{a}_n\) while `step` stays `&self`.

## Files

Create:

- `week2/md/src/system.rs` — `System`, `compute_accelerations`, kinetic / potential / total energy
- `week2/md/src/integrators.rs` — trait, `advance`, `Euler`, `VelocityVerlet`, `relative_energy_errors`
- `week2/md/tests/dimer.rs` — conservation test (integration test, public API only)
- `week2/md/examples/dimer.rs` — error plot (VERIFY, last)

Modify:

- `week2/md/src/lib.rs` — `mod system; mod integrators;` and `pub use` of the public API. Do not move or rewrite existing `energy` / `force` / greeting tests.

Do not modify:

- `week2/md/src/main.rs`
- `week2/md/examples/field.rs`
- pair formulas in `energy` / `force`

The dimer test lives in `tests/dimer.rs` so the red commit can add the test without making `cargo test --lib` fail to compile. Existing `greeting` / `well_depth` / `force_matches_energy_derivative` tests stay green on `--lib` throughout.

## Public API

```rust
pub struct System {
    pub positions: Vec<[f64; 2]>,
    pub velocities: Vec<[f64; 2]>,
    pub accelerations: Vec<[f64; 2]>,
}

impl System {
    pub fn new(positions: Vec<[f64; 2]>, velocities: Vec<[f64; 2]>) -> Self;
    pub fn n_atoms(&self) -> usize;
}

pub fn compute_accelerations(system: &mut System);

pub fn kinetic_energy(system: &System) -> f64;
pub fn potential_energy(system: &System) -> f64;
pub fn total_energy(system: &System) -> f64;

pub trait Integrator {
    fn step(&self, system: &mut System, dt: f64);
}

pub fn advance(method: &impl Integrator, system: &mut System, dt: f64);

pub struct Euler;
pub struct VelocityVerlet;

pub fn relative_energy_errors(
    method: &impl Integrator,
    system: &mut System,
    dt: f64,
    n_steps: usize,
) -> Vec<f64>;
```

`System::new` asserts `positions.len() == velocities.len()` and fills `accelerations` with zeros of that length. Derive `Clone` so the test and plot can copy the initial state.

`advance` is exactly `method.step(system, dt)`.

## Forces (`compute_accelerations`)

Open boundaries, plain Lennard-Jones, all pairs, no cutoff.

1. Zero every stored acceleration.
2. For each pair \(i < j\):
   - \(\vec{d} = \vec{x}_i - \vec{x}_j\)
   - \(r = |\vec{d}|\)
   - \(\vec{F}_i = \mathrm{force}(r)\,\vec{d}/r\)
   - \(\vec{F}_j = -\vec{F}_i\)
   - \(\vec{a} = \vec{F}\) because \(m = 1\); accumulate into `accelerations[i]` and `accelerations[j]`.
3. Call the existing scalar `force(r)`. Never re-derive it from `energy`.

Assume \(r > 0\) (the dimer oscillates near \(r_0 \approx 1.122\)).

Accelerations are **not** computed once and frozen.

- **Before the first step** of a run: `compute_accelerations` from the initial positions (so Verlet has \(\vec{a}_n\)).
- **Euler, every step:** recompute \(\vec{a}_n\) from the **current** positions at the start of `step`, then \(\vec{x} \leftarrow \vec{x}+\vec{v}\Delta t\), \(\vec{v} \leftarrow \vec{v}+\vec{a}\Delta t\). If Euler reused the initial \(\vec{a}\) forever, the conservation test would be meaningless.
- **Verlet, every step:** half-kick with stored \(\vec{a}_n\); drift \(\vec{x}\) with the half-step velocity; `compute_accelerations` at the new \(\vec{x}\) (the step’s **one** force evaluation → \(\vec{a}_{n+1}\)); half-kick with \(\vec{a}_{n+1}\); leave \(\vec{a}_{n+1}\) stored for the next step.

## Integrators

Forward Euler (old state for both updates; **not** Euler-Cromer):

```text
compute_accelerations(system)          # a_n from current x
x += v * dt
v += a * dt
```

Velocity-Verlet (sheet algorithm):

```text
v += 0.5 * dt * a_n                   # stored accelerations
x += dt * v                           # drift with half-step v
compute_accelerations(system)          # a_{n+1} from new x
v += 0.5 * dt * a_{n+1}
```

## Energy measurement

\[
E = \tfrac12\sum_i |\vec{v}_i|^2 + \sum_{i<j} U(r_{ij})
\]

For the dimer this is \(\tfrac12(|\vec{v}_0|^2+|\vec{v}_1|^2)+U(r)\).

`relative_energy_errors`:

1. `e0 = total_energy(system)` **before** any step. For the dimer at rest this equals `energy(1.2)` = \(U(1.2)\). Do not hard-code `1.2` inside this function.
2. `compute_accelerations(system)` once (required for Verlet; Euler will recompute again at the start of each step).
3. For `k in 0..n_steps`: `advance(method, system, dt)`; push \((E - e_0)/|e_0|\) where \(E\) is `total_energy` after that step.

Return length is exactly `n_steps`. Index `499` is the error after step 500 when `n_steps = 500`. The series does **not** include the \(t=0\) point (that error is identically 0). The plot example may prepend `(t=0, err=0)` for display only; it must still call this function rather than duplicating the time loop.

The test and the plot must both call `relative_energy_errors`. They must not each write their own `advance` loop.

## Dimer test contract

File: `week2/md/tests/dimer.rs`. One test function runs **both** methods. The only difference between the two runs is which integrator value is passed into `relative_energy_errors` (and therefore into `advance`). Do not call differently named step functions.

Numbers belong in the test body, not only in comments:

- positions `(0.0, 0.0)` and `(1.2, 0.0)`; both velocities `(0.0, 0.0)`
- `dt = 0.01`, `n = 500` for **both** methods
- Verlet: \(\max_k |\mathrm{err}[k]| < 10^{-3}\)
- Euler: \(|\mathrm{err}[499]| > 0.5\)

Do not run 5000 Verlet steps in this test.

## Plot (VERIFY, last)

`week2/md/examples/dimer.rs`, same pattern as `field.rs`:

- Left: Euler and Verlet, 500 steps, `dt = 0.01`, relative error vs time on \([0, 5]\)
- Right: Verlet only, 5000 steps, error × 1000 vs time on \([0, 50]\)
- Write `week2/dimer.png` via `CARGO_MANIFEST_DIR/../dimer.png`
- Reproduce from `week2/`: `cargo run --manifest-path md/Cargo.toml --example dimer`

Use `plotters` (already a dev-dependency). Implement this only after the dimer test is green. Own commit, not shared with the red test.

## Testing / TDD order

1. Write `tests/dimer.rs` using the public API above.
2. Watch it fail: `cargo test --lib` still passes (3 existing tests); `cargo test` fails to compile the integration test because `System` / `Integrator` do not exist yet.
3. Commit **only** that failing test.
4. Then implement `System`, forces, trait, both integrators, and `relative_energy_errors` until `cargo test --manifest-path md/Cargo.toml` is all green.
5. Then add the plot example.

No production integrator code until the dimer test exists, fails, and is committed.

## Git

- Commit the failing dimer test **before** any `System` / `Integrator` implementation that would make it pass.
- Keep greeting / well_depth / force tests green on `cargo test --lib`.
- Separate commits (red test, green library, plot).
- Do not `git add .`.
- Do not commit PDFs, `target/`, `.DS_Store`, or the week-2 learning sheet.
- Do not push unless asked.

## Success

- `cargo test --manifest-path week2/md/Cargo.toml` is all green and includes a dimer test that runs both integrators from the same initial state through the trait.
- On that state, Verlet’s max \(|err| < 10^{-3}\) and Euler’s \(|err[499]| > 0.5\).
- `examples/dimer.rs` writes `week2/dimer.png` after the test is green.
