# Cell List Forces and Heating (Part 5)

Date: 2026-09-16
Status: locked by the Week 2 remaining-finish plan (ForceMode on System / RunParams)
Crate: `week2/md`
Depends on: Part 4 periodic LJ fluid (`Interaction::Periodic`, `VelocityVerlet`, CLI)

## Goal

Speed up periodic force and energy evaluation with a cell list that reproduces naive all-pairs results within rounding, keep the original loop behind `--force naive`, default to `--force cells`, and add `--ramp-to` so a production thermostat target rises linearly. Integrators stay unchanged. The Part 3 dimer and Part 4 contract checks stay green.

## Non-goals

- Do not change `Integrator::step(&self, system: &mut System, dt: f64)`.
- Do not wrap positions inside Verlet.
- Do not reimplement or finite-difference `energy(r)` / `force(r)`.
- Do not put a cell list on `Interaction::Open` (dimer stays all-pairs).
- Do not heat the default `artifacts/` contract run.
- Do not commit lecture PDFs, `artifacts/`, `target/`, or `week2-sim.py`.

## Architecture (locked)

`ForceMode { Naive, Cells }` lives on `System` and `RunParams`. `compute_accelerations` and `potential_energy` branch on it for `Interaction::Periodic`. `VelocityVerlet::step` still only calls `compute_accelerations`. The run loop still wraps after each step. Heating rescales only during production when `ramp_to` is set.

| Unit | Responsibility |
|------|----------------|
| `ForceMode` | `Naive` all-pairs vs `Cells` neighbour list |
| `System.force_mode` | What the current force/energy path uses |
| Cell list rebuild | Every force or energy evaluation from current positions |
| `RunParams.force` / `--force` | CLI switch; default `cells` |
| `RunParams.ramp_to` / `--ramp-to` | Optional linear production thermostat |
| `ramp_target` | \(T(s) = T_0 + (T_1-T_0)\,s/N_\mathrm{steps}\) |

## Cell rule

For a periodic box \((L_x, L_y)\) and cutoff \(r_c\):

\[
n_x = \lfloor L_x / r_c \rfloor,\quad
w_x = L_x / n_x
\]

and likewise \(n_y\), \(w_y\). If \(n_x < 1\) or \(n_y < 1\), use the naive path (a box thinner than \(r_c\) cannot host a valid cell grid).

Particle \(i\) is binned by

\[
c_x = \min\bigl(n_x-1,\ \lfloor x_i / w_x \rfloor\bigr)
\]

(and \(c_y\)). Search the nine cells \((c_x+\delta_x,\ c_y+\delta_y)\) for \(\delta\in\{-1,0,1\}\), wrapping with Euclidean remainder. Deduplicate the resulting cell indices before visiting particles so a two-cell-wide box does not apply a pair twice. Each unordered pair is applied once (\(j > i\)). Pair vectors stay minimum-image via the existing `pair_vector`.

Rebuild the bins from current coordinates on every call. Do not maintain an incremental list across steps.

## Forces and energy

Periodic pair physics is unchanged: force \(F(r)\) for \(r < r_c\), energy \(U(r)-U(r_c)\) for \(r < r_c\), both zero at and beyond \(r_c\). Naive and cells must agree on accelerations and potential energy within a rounding tolerance (`1e-12` relative to `1.0` or the magnitude).

`md check` reconstructs `System::periodic` from saved frames. Default `force_mode` on that constructor is `Cells`. Because the two paths agree, stored `E_pot` still matches the recompute.

## Heating

`--ramp-to T1` is optional. Equilibration still rescales to `--temperature` every 50 eq steps. During production, every 50 production steps (and using the production step index \(s=1\ldots N\)):

\[
T(s) = T_0 + (T_1 - T_0)\, s / N
\]

so \(s=0\) would be \(T_0\) and the last step is \(T_1\). After each production rescale, recompute accelerations so Verlet's stored \(a\) matches the new velocities. `run.json` records `ramp_to` (`null` when the flag is absent). Heating output goes to a separate `--out` (for example `docs/`), never the unheated `artifacts/` contract.

## CLI

```
md run --force cells|naive   # default cells
md run --ramp-to 1.2         # optional
```

`run.json` gains `ramp_to`. Existing keys stay.

## Tests

- Perturbed triangular lattice: naive vs cells accelerations and energy.
- Pair across a periodic boundary.
- Pair at cutoff (just inside vs just outside).
- Two-cell-wide box (dedupe).
- `ramp_target` endpoints and midpoint; a short `md run --ramp-to` writes `ramp_to` in `run.json`.
- Existing dimer, pbc, and CLI tests remain green.

## Error handling

Unknown `--force` values are clap parse errors. `--ramp-to` must be finite and positive when present. Cell construction never panics on a valid periodic `System`; it falls back to naive if \(n_x\) or \(n_y\) is zero.
