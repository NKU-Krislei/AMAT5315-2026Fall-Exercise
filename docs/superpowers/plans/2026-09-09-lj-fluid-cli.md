# LJ Fluid CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add periodic shifted-LJ forces, a contract fluid run that writes JSON, `md check` / `md video`, and `make reproduce`, without breaking the Part 3 dimer.

**Architecture:** `Interaction` on `System` selects Open vs Periodic. Integrators stay Part 3. The run loop wraps positions, thermostats during equilibration, and writes frames. Check recomputes from raw `pos`/`vel`.

**Tech Stack:** Rust `md` crate, `clap` 4 derive, `serde`/`serde_json`, `rand` 0.8, `plotters`, system `ffmpeg`.

## Global Constraints

- Reuse `energy(r)` / `force(r)`; do not reimplement or finite-difference the force.
- `Integrator::step(&self, system: &mut System, dt: f64)` unchanged. Do not wrap inside Verlet.
- `System::new` remains Open-boundary so `tests/dimer.rs` stays valid.
- Open: full LJ. Periodic: min-image, \(r_c=2.5\), \(U_\mathrm{cut}=U(r)-U(r_c)\) for \(r<r_c\).
- Contract defaults: `n=100 rho=0.8 temperature=0.5 dt=0.01 eq-steps=2000 steps=10000 sample-every=50 seed=2026 out=artifacts integrator=velocity-verlet`.
- Four tests first in spirit: force sum, cutoff from **inside** \(r_c\), binary writes frames, contract physics in **release**.
- Dimer / well_depth / force tests stay green on `cargo test --lib`.
- Separate commits. No `git add .`. No PDFs, `target/`, `artifacts/`. No push unless asked.
- `md video` last. Commands from repo root unless noted.

## File structure

- Modify: `week2/md/src/system.rs` — `Interaction`, min-image, shifted U, wrap
- Create: `week2/md/src/lattice.rs` — triangular lattice, Maxwell-Boltzmann, thermostat
- Create: `week2/md/src/simulate.rs` — run loop + JSON IO
- Create: `week2/md/src/check.rs` — drift, T, chi²
- Create: `week2/md/src/rdf.rs` — \(g(r)\)
- Create: `week2/md/src/video.rs` — frames + ffmpeg
- Modify: `week2/md/src/lib.rs` — modules / `pub use`
- Modify: `week2/md/src/main.rs` — clap
- Modify: `week2/md/Cargo.toml` — deps
- Create: `week2/md/tests/pbc.rs`, `week2/md/tests/cli.rs`
- Create: `week2/Makefile`, `week2/.gitignore`

---

### Task 1: Failing PBC force and cutoff tests

**Files:**
- Create: `week2/md/tests/pbc.rs`

**Interfaces:**
- Consumes: wished-for `System::periodic`, `compute_accelerations`, `potential_energy`, `energy`
- Produces: two failing tests

- [ ] **Step 1: Write the tests**

Create `week2/md/tests/pbc.rs`:

```rust
use md::{compute_accelerations, energy, potential_energy, System};

const RC: f64 = 2.5;

#[test]
fn periodic_forces_sum_to_zero() {
    let positions = vec![
        [1.0, 1.0],
        [3.5, 2.0],
        [6.0, 5.0],
        [8.0, 8.0],
        [2.0, 9.0],
    ];
    let velocities = vec![[0.0, 0.0]; 5];
    let mut sys = System::periodic(positions, velocities, [12.0, 12.0], RC);
    compute_accelerations(&mut sys);
    let mut fx = 0.0;
    let mut fy = 0.0;
    for a in &sys.accelerations {
        fx += a[0];
        fy += a[1];
    }
    assert!(fx.abs() < 1e-12, "sum Fx = {fx}");
    assert!(fy.abs() < 1e-12, "sum Fy = {fy}");
}

#[test]
fn cutoff_potential_continuous_from_inside() {
    let r_in = RC - 1e-4;
    let mut sys = System::periodic(
        vec![[0.0, 0.0], [r_in, 0.0]],
        vec![[0.0, 0.0], [0.0, 0.0]],
        [20.0, 20.0],
        RC,
    );
    let u_in = potential_energy(&sys);
    let expected = energy(r_in) - energy(RC);
    assert!(
        (u_in - expected).abs() < 1e-12,
        "inside: U={u_in} expected={expected}"
    );
    assert!(
        energy(r_in).abs() > 1e-5,
        "unshifted U(r) must not already be ~0, else the shift is untested"
    );

    sys.positions[1][0] = RC + 1e-4;
    let u_out = potential_energy(&sys);
    assert!(u_out.abs() < 1e-15, "outside cutoff U={u_out}");
}
```

- [ ] **Step 2: Confirm `--lib` still passes**

Run: `cargo test --manifest-path week2/md/Cargo.toml --lib`

Expected: existing 3 lib tests passed.

- [ ] **Step 3: Watch PBC tests fail to compile**

Run: `cargo test --manifest-path week2/md/Cargo.toml --test pbc`

Expected: compile error, no `System::periodic`. Do not stub it in this commit.

- [ ] **Step 4: Commit only the failing tests**

```bash
git add week2/md/tests/pbc.rs
git commit -m "$(cat <<'EOF'
Add failing periodic force-sum and cutoff tests.

EOF
)"
```

---

### Task 2: Interaction, min-image, shifted energy

**Files:**
- Modify: `week2/md/src/system.rs`
- Modify: `week2/md/src/lib.rs` (re-export `Interaction` if public; `System::periodic` is enough)

**Interfaces:**
- Consumes: `energy`, `force`
- Produces: `Interaction`, `System::periodic`, wrap-aware energy/forces; Open path identical to Part 3

- [ ] **Step 1: Replace `week2/md/src/system.rs` with**

```rust
use crate::{energy, force};

#[derive(Clone, Debug)]
pub enum Interaction {
    Open,
    Periodic { box_xy: [f64; 2], rc: f64 },
}

#[derive(Clone, Debug)]
pub struct System {
    pub positions: Vec<[f64; 2]>,
    pub velocities: Vec<[f64; 2]>,
    pub accelerations: Vec<[f64; 2]>,
    pub interaction: Interaction,
}

impl System {
    pub fn new(positions: Vec<[f64; 2]>, velocities: Vec<[f64; 2]>) -> Self {
        assert_eq!(positions.len(), velocities.len());
        let accelerations = vec![[0.0, 0.0]; positions.len()];
        Self {
            positions,
            velocities,
            accelerations,
            interaction: Interaction::Open,
        }
    }

    pub fn periodic(
        positions: Vec<[f64; 2]>,
        velocities: Vec<[f64; 2]>,
        box_xy: [f64; 2],
        rc: f64,
    ) -> Self {
        let mut sys = Self::new(positions, velocities);
        sys.interaction = Interaction::Periodic { box_xy, rc };
        sys
    }

    pub fn n_atoms(&self) -> usize {
        self.positions.len()
    }
}

fn min_image(d: f64, length: f64) -> f64 {
    d - length * (d / length).round()
}

fn pair_vector(system: &System, i: usize, j: usize) -> (f64, f64, f64) {
    let mut dx = system.positions[i][0] - system.positions[j][0];
    let mut dy = system.positions[i][1] - system.positions[j][1];
    if let Interaction::Periodic { box_xy, .. } = system.interaction {
        dx = min_image(dx, box_xy[0]);
        dy = min_image(dy, box_xy[1]);
    }
    let r = (dx * dx + dy * dy).sqrt();
    (dx, dy, r)
}

pub fn wrap_positions(system: &mut System) {
    let Interaction::Periodic { box_xy, .. } = system.interaction else {
        return;
    };
    for p in &mut system.positions {
        p[0] = p[0].rem_euclid(box_xy[0]);
        p[1] = p[1].rem_euclid(box_xy[1]);
    }
}

pub fn compute_accelerations(system: &mut System) {
    let n = system.n_atoms();
    for a in &mut system.accelerations {
        *a = [0.0, 0.0];
    }
    let rc = match system.interaction {
        Interaction::Open => f64::INFINITY,
        Interaction::Periodic { rc, .. } => rc,
    };
    for i in 0..n {
        for j in (i + 1)..n {
            let (dx, dy, r) = pair_vector(system, i, j);
            if r >= rc || r == 0.0 {
                continue;
            }
            let f = force(r);
            let fx = f * dx / r;
            let fy = f * dy / r;
            system.accelerations[i][0] += fx;
            system.accelerations[i][1] += fy;
            system.accelerations[j][0] -= fx;
            system.accelerations[j][1] -= fy;
        }
    }
}

pub fn kinetic_energy(system: &System) -> f64 {
    system
        .velocities
        .iter()
        .map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1]))
        .sum()
}

pub fn potential_energy(system: &System) -> f64 {
    let n = system.n_atoms();
    let (rc, shift) = match system.interaction {
        Interaction::Open => (f64::INFINITY, 0.0),
        Interaction::Periodic { rc, .. } => (rc, energy(rc)),
    };
    let mut u = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let (_, _, r) = pair_vector(system, i, j);
            if r >= rc {
                continue;
            }
            u += energy(r) - shift;
        }
    }
    u
}

pub fn total_energy(system: &System) -> f64 {
    kinetic_energy(system) + potential_energy(system)
}
```

Re-export `wrap_positions` and `Interaction` from `lib.rs` next to the existing `system` exports.

- [ ] **Step 2: Run tests**

`cargo test --manifest-path week2/md/Cargo.toml --lib` → 3 passed.

`cargo test --manifest-path week2/md/Cargo.toml --test pbc` → 2 passed.

`cargo test --manifest-path week2/md/Cargo.toml --test dimer` → 1 passed.

If dimer fails, the Open branch is wrong; fix without changing `tests/dimer.rs`.

- [ ] **Step 3: Commit**

```bash
git add week2/md/src/system.rs week2/md/src/lib.rs
git commit -m "$(cat <<'EOF'
Add periodic min-image forces and a shifted cutoff.

EOF
)"
```

---

### Task 3: Failing CLI / trajectory IO test

**Files:**
- Create: `week2/md/tests/cli.rs`

**Interfaces:**
- Consumes: binary `md run` (does not exist as subcommand yet)
- Produces: failing test `run_writes_run_json_and_traj_jsonl`

- [ ] **Step 1: Write `week2/md/tests/cli.rs`**

```rust
use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn tmp_out() -> PathBuf {
    let p = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("cli_io");
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn run_writes_run_json_and_traj_jsonl() {
    let out = tmp_out();
    let exe = env!("CARGO_BIN_EXE_md");
    let status = Command::new(exe)
        .args([
            "run",
            "--n",
            "36",
            "--rho",
            "0.8",
            "--temperature",
            "0.5",
            "--dt",
            "0.01",
            "--eq-steps",
            "50",
            "--steps",
            "50",
            "--sample-every",
            "50",
            "--seed",
            "2026",
            "--out",
        ])
        .arg(&out)
        .status()
        .expect("spawn md");
    assert!(status.success(), "md run failed: {status}");

    let run: Value =
        serde_json::from_str(&fs::read_to_string(out.join("run.json")).unwrap()).unwrap();
    for key in [
        "n",
        "rho",
        "box",
        "dt",
        "temperature",
        "eq_steps",
        "steps",
        "sample_every",
        "seed",
        "integrator",
    ] {
        assert!(run.get(key).is_some(), "missing {key}");
    }
    assert_eq!(run["n"], 36);
    assert_eq!(run["integrator"], "velocity-verlet");

    let traj = fs::read_to_string(out.join("traj.jsonl")).unwrap();
    let frames: Vec<Value> = traj
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0]["step"], 50);
    assert!(frames[0]["t"].as_f64().unwrap() > 0.0);
    assert!(frames[0].get("pos").is_some());
    assert!(frames[0].get("vel").is_some());
    assert!(frames[0].get("E_pot").is_some());
    assert!(frames[0].get("E_kin").is_some());
}
```

Add `serde_json` to `[dev-dependencies]` **in this task only if** the test must compile before the run implementation. Prefer: this test will fail to compile on `CARGO_BIN_EXE` still working but `run` subcommand missing — that is an assertion failure / clap error, which is a valid red. Add `serde_json` to `[dependencies]` in Task 4 with clap/serde; for Task 3, add it to dev-dependencies so the test crate compiles.

Cargo.toml after this step:

```toml
[dev-dependencies]
plotters = "0.3"
serde_json = "1"
```

- [ ] **Step 2: Run the test — expect failure** (binary still Hello World / unknown subcommand `run`)

`cargo test --manifest-path week2/md/Cargo.toml --test cli`

Expected: FAIL (nonzero exit from `md` or missing files). `--lib` and dimer still pass.

- [ ] **Step 3: Commit**

```bash
git add week2/md/tests/cli.rs week2/md/Cargo.toml week2/md/Cargo.lock
git commit -m "$(cat <<'EOF'
Add failing CLI test for run.json and traj.jsonl.

EOF
)"
```

---

### Task 4: Lattice, thermostat, run loop, `clap run`

**Files:**
- Create: `week2/md/src/lattice.rs`
- Create: `week2/md/src/simulate.rs`
- Modify: `week2/md/src/lib.rs`, `week2/md/src/main.rs`, `week2/md/Cargo.toml`

**Interfaces:**
- Consumes: `System::periodic`, `advance`, `VelocityVerlet`, `wrap_positions`, `potential_energy`, `kinetic_energy`
- Produces: `run_simulation(params, out_dir)`; CLI `md run`

- [ ] **Step 1: Cargo.toml `[dependencies]`**

```toml
[dependencies]
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rand = { version = "0.8", features = ["std", "std_rng"] }
```

Keep `plotters` in dev-dependencies until the video task.

- [ ] **Step 2: Implement lattice + thermostat + simulate + clap** so `run_writes_run_json_and_traj_jsonl` passes.

Create `week2/md/src/lattice.rs`:

```rust
use crate::system::{kinetic_energy, System};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

pub fn triangular_lattice(n: usize, rho: f64) -> (Vec<[f64; 2]>, [f64; 2]) {
    let n_side = (n as f64).sqrt().round() as usize;
    assert_eq!(n_side * n_side, n, "n must be a square");
    let a = (2.0 / (3.0_f64.sqrt() * rho)).sqrt();
    let h = 3.0_f64.sqrt() / 2.0 * a;
    let box_xy = [n_side as f64 * a, n_side as f64 * h];
    let mut positions = Vec::with_capacity(n);
    for j in 0..n_side {
        for i in 0..n_side {
            let x = (i as f64 + 0.5 * (j % 2) as f64) * a;
            let y = j as f64 * h;
            positions.push([x, y]);
        }
    }
    (positions, box_xy)
}

fn unit_gaussian(rng: &mut StdRng) -> f64 {
    let u1 = rng.gen::<f64>().max(1e-12);
    let u2 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

pub fn seed_maxwell_boltzmann(system: &mut System, temperature: f64, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let s = temperature.sqrt();
    for v in &mut system.velocities {
        v[0] = s * unit_gaussian(&mut rng);
        v[1] = s * unit_gaussian(&mut rng);
    }
    let n = system.n_atoms() as f64;
    let mut mx = 0.0;
    let mut my = 0.0;
    for v in &system.velocities {
        mx += v[0];
        my += v[1];
    }
    mx /= n;
    my /= n;
    for v in &mut system.velocities {
        v[0] -= mx;
        v[1] -= my;
    }
    rescale_temperature(system, temperature);
}

pub fn rescale_temperature(system: &mut System, temperature: f64) {
    let n = system.n_atoms();
    assert!(n > 1);
    let t_thermo = 2.0 * kinetic_energy(system) / (2 * n - 2) as f64;
    let factor = (temperature / t_thermo).sqrt();
    for v in &mut system.velocities {
        v[0] *= factor;
        v[1] *= factor;
    }
}
```

Create `week2/md/src/simulate.rs`:

```rust
use crate::integrators::{advance, VelocityVerlet};
use crate::lattice::{rescale_temperature, seed_maxwell_boltzmann, triangular_lattice};
use crate::system::{
    compute_accelerations, kinetic_energy, potential_energy, wrap_positions, System,
};
use serde::Serialize;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

pub const RC: f64 = 2.5;
pub const EQ_RESCALE_EVERY: usize = 50;

#[derive(Clone, Debug)]
pub struct RunParams {
    pub n: usize,
    pub rho: f64,
    pub temperature: f64,
    pub dt: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
}

impl Default for RunParams {
    fn default() -> Self {
        Self {
            n: 100,
            rho: 0.8,
            temperature: 0.5,
            dt: 0.01,
            eq_steps: 2000,
            steps: 10000,
            sample_every: 50,
            seed: 2026,
        }
    }
}

#[derive(Serialize)]
struct RunJson {
    n: usize,
    rho: f64,
    #[serde(rename = "box")]
    box_xy: [f64; 2],
    dt: f64,
    temperature: f64,
    eq_steps: usize,
    steps: usize,
    sample_every: usize,
    seed: u64,
    integrator: &'static str,
}

#[derive(Serialize)]
struct Frame {
    step: usize,
    t: f64,
    pos: Vec<[f64; 2]>,
    vel: Vec<[f64; 2]>,
    #[serde(rename = "E_pot")]
    e_pot: f64,
    #[serde(rename = "E_kin")]
    e_kin: f64,
}

pub fn run_simulation(params: &RunParams, out_dir: &Path) -> Result<(), String> {
    fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let (positions, box_xy) = triangular_lattice(params.n, params.rho);
    let velocities = vec![[0.0, 0.0]; params.n];
    let mut system = System::periodic(positions, velocities, box_xy, RC);
    seed_maxwell_boltzmann(&mut system, params.temperature, params.seed);
    compute_accelerations(&mut system);

    let integrator = VelocityVerlet;
    for step in 1..=params.eq_steps {
        advance(&integrator, &mut system, params.dt);
        wrap_positions(&mut system);
        if step % EQ_RESCALE_EVERY == 0 {
            rescale_temperature(&mut system, params.temperature);
            compute_accelerations(&mut system);
        }
    }

    let run = RunJson {
        n: params.n,
        rho: params.rho,
        box_xy,
        dt: params.dt,
        temperature: params.temperature,
        eq_steps: params.eq_steps,
        steps: params.steps,
        sample_every: params.sample_every,
        seed: params.seed,
        integrator: "velocity-verlet",
    };
    fs::write(
        out_dir.join("run.json"),
        serde_json::to_string_pretty(&run).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    let mut traj = BufWriter::new(File::create(out_dir.join("traj.jsonl")).map_err(|e| e.to_string())?);
    for step in 1..=params.steps {
        advance(&integrator, &mut system, params.dt);
        wrap_positions(&mut system);
        if step % params.sample_every == 0 {
            let frame = Frame {
                step,
                t: step as f64 * params.dt,
                pos: system.positions.clone(),
                vel: system.velocities.clone(),
                e_pot: potential_energy(&system),
                e_kin: kinetic_energy(&system),
            };
            writeln!(traj, "{}", serde_json::to_string(&frame).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}
```

After a thermostat rescale, accelerations must be refreshed because Verlet’s stored \(\vec{a}\) is from pre-rescale positions (positions did not change; velocities did — accelerations depend only on positions, so the extra `compute_accelerations` is optional but cheap insurance). Keep it.

`lib.rs` add `mod lattice; mod simulate;` and `pub use simulate::{run_simulation, RunParams};`.

Replace `week2/md/src/main.rs`:

```rust
use clap::{Parser, Subcommand};
use md::{run_simulation, RunParams};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "md")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Run {
        #[arg(long, default_value_t = 100)]
        n: usize,
        #[arg(long, default_value_t = 0.8)]
        rho: f64,
        #[arg(long, default_value_t = 0.5)]
        temperature: f64,
        #[arg(long, default_value_t = 0.01)]
        dt: f64,
        #[arg(long, default_value_t = 2000)]
        eq_steps: usize,
        #[arg(long, default_value_t = 10000)]
        steps: usize,
        #[arg(long, default_value_t = 50)]
        sample_every: usize,
        #[arg(long, default_value_t = 2026)]
        seed: u64,
        #[arg(long, default_value = "artifacts")]
        out: PathBuf,
    },
    Check { dir: PathBuf },
    Video {
        dir: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

fn main() -> ExitCode {
    match Cli::parse().command {
        Commands::Run {
            n,
            rho,
            temperature,
            dt,
            eq_steps,
            steps,
            sample_every,
            seed,
            out,
        } => {
            let params = RunParams {
                n,
                rho,
                temperature,
                dt,
                eq_steps,
                steps,
                sample_every,
                seed,
            };
            if let Err(e) = run_simulation(&params, &out) {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }
        Commands::Check { .. } => {
            eprintln!("md check is not implemented yet");
            ExitCode::FAILURE
        }
        Commands::Video { .. } => {
            eprintln!("md video is not implemented yet");
            ExitCode::FAILURE
        }
    }
}
```

- [ ] **Step 3: Verify**

`cargo test --manifest-path week2/md/Cargo.toml --test cli` → pass.

`cargo test --manifest-path week2/md/Cargo.toml --test dimer` → pass.

`cargo test --manifest-path week2/md/Cargo.toml --test pbc` → pass.

- [ ] **Step 4: Commit**

```bash
git add week2/md/Cargo.toml week2/md/Cargo.lock week2/md/src/lattice.rs week2/md/src/simulate.rs week2/md/src/lib.rs week2/md/src/main.rs
git commit -m "$(cat <<'EOF'
Run a periodic LJ fluid and write trajectory JSON.

EOF
)"
```

---

### Task 5: `md check` and contract physics test

**Files:**
- Create: `week2/md/src/check.rs`
- Modify: `week2/md/src/main.rs`, `week2/md/src/lib.rs`
- Modify: `week2/md/tests/cli.rs` (add contract test)

**Interfaces:**
- Consumes: traj frames, `System::periodic`, `potential_energy`, `kinetic_energy`
- Produces: `check_artifacts(dir) -> CheckReport` with drift, `T_speed`, `chi2`; CLI exit 0 iff PASS

- [ ] **Step 1: Write `contract_run_passes_physics` in `tests/cli.rs`**

```rust
#[test]
fn contract_run_passes_physics() {
    if cfg!(debug_assertions) {
        eprintln!("skipping contract run in debug; cargo test --release runs it");
        return;
    }
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("contract");
    let _ = fs::remove_dir_all(&out);
    let exe = env!("CARGO_BIN_EXE_md");
    let status = Command::new(exe)
        .args(["run", "--out"])
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let status = Command::new(exe).arg("check").arg(&out).status().unwrap();
    assert!(status.success(), "md check should PASS on the contract run");
}
```

In debug this returns immediately (still “pass”). In release it must run the real defaults. Watch **release** fail until check exists: run `cargo test --manifest-path week2/md/Cargo.toml --release --test cli -- contract_run_passes_physics`.

- [ ] **Step 2: Implement check**

Recompute energies from pos/vel with the box and `rc=2.5` from `run.json`. Fail if stored `E_pot`/`E_kin` disagree (abs or rel `1e-9`). Drift with \(k=\max(1,\lfloor n_f/10\rfloor)\). \(T_\mathrm{speed}=\langle v^2\rangle/2\). Chi-squared 24 bins as in the spec. Print three lines + `PASS`/`FAIL`.

Wire `md check <dir>`.

- [ ] **Step 3: Verify**

`cargo test --manifest-path week2/md/Cargo.toml --release --test cli` → both CLI tests pass.

`cargo test --manifest-path week2/md/Cargo.toml --test dimer` → pass.

- [ ] **Step 4: Commit**

```bash
git add week2/md/src/check.rs week2/md/src/lib.rs week2/md/src/main.rs week2/md/tests/cli.rs
git commit -m "$(cat <<'EOF'
Check fluid trajectories for drift, temperature, and speed shape.

EOF
)"
```

---

### Task 6: `make reproduce` and gitignore

**Files:**
- Create: `week2/Makefile`
- Create: `week2/.gitignore`

- [ ] **Step 1: Write files**

`week2/Makefile`:

```make
.PHONY: reproduce
reproduce:
	cargo run --manifest-path md/Cargo.toml --release -- run --out artifacts
```

`week2/.gitignore`:

```
artifacts/
*.mp4
```

- [ ] **Step 2: Commit**

```bash
git add week2/Makefile week2/.gitignore
git commit -m "$(cat <<'EOF'
Add make reproduce for the default fluid run.

EOF
)"
```

Do not add `artifacts/`.

---

### Task 7: `md video` (VERIFY, last)

**Files:**
- Create: `week2/md/src/rdf.rs`
- Create: `week2/md/src/video.rs`
- Modify: `week2/md/src/main.rs`, `week2/md/src/lib.rs`, `week2/md/Cargo.toml` (move/add `plotters` under `[dependencies]`)

**Interfaces:**
- Consumes: `traj.jsonl`, `run.json`
- Produces: MP4 `< 2 MB` via ffmpeg

- [ ] **Step 1: Implement \(g(r)\) and video after check is green**

Min-image distances, rings to `0.5 * min(Lx,Ly)`, \(\Delta r \approx 0.05\). Left: circles at wrapped positions. Right: line plot of \(g(r)\). Temp PNGs in a unique temp dir. `ffmpeg -y -framerate 20 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 28` (adjust to stay under 2 MB). If ffmpeg missing, error with install hint.

- [ ] **Step 2: Manual run from `week2/` after a reproduce (or a short run)**

```bash
cargo run --manifest-path md/Cargo.toml --release -- video artifacts --out fluid.mp4
```

Expected: `fluid.mp4` exists, size < 2_000_000 bytes. Do not `git add` the mp4.

`cargo test --manifest-path week2/md/Cargo.toml --release` still all green.

- [ ] **Step 3: Commit source only**

```bash
git add week2/md/src/rdf.rs week2/md/src/video.rs week2/md/src/main.rs week2/md/src/lib.rs week2/md/Cargo.toml week2/md/Cargo.lock
git commit -m "$(cat <<'EOF'
Render fluid video with atoms and g(r).

EOF
)"
```
