"""Hand-written forward- and reverse-mode AD of the Lennard-Jones pair energy.

Wengert list (r = 1.3, JAX, 64-bit floats):

    a = r**-6      # node 1
    b = a**2       # node 2
    c = b - a      # node 3
    U = 4.0 * c    # node 4

Forward mode seeds the tangent of r with 1 and pushes tangents through the
nodes in order. Reverse mode seeds the adjoint of U with 1 and pulls adjoints
back through the nodes in reverse order. Both passes are written out node by
node with the local (partial) derivatives spelled out in the comments; JAX's
own ``jax.grad`` is computed afterwards as a cross-check, alongside the
closed-form dU/dr = 24 r^-7 - 48 r^-13.

Writes ``artifacts/ad/derivatives.json`` (paths relative to ``week5/``).
"""

import json
import math
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # 64-bit floats
import jax.numpy as jnp  # noqa: E402

R = 1.3

# ------------------------------------------------------------- primal pass
r = jnp.asarray(R, dtype=jnp.float64)
a = r**-6  # node 1
b = a**2  # node 2
c = b - a  # node 3
U = 4.0 * c  # node 4

# ------------------------------------------- forward mode, seed dr_tan = 1
r_tan = jnp.asarray(1.0, dtype=jnp.float64)  # dU/dr with seed 1
a_tan = -6.0 * r**-7 * r_tan  # node 1: da/dr = -6 r^-7
b_tan = 2.0 * a * a_tan  # node 2: db/da = 2a
c_tan = b_tan - a_tan  # node 3: dc/db = 1, dc/da = -1
U_tan = 4.0 * c_tan  # node 4: dU/dc = 4

# ------------------------------------------- reverse mode, seed U_adj = 1
U_adj = jnp.asarray(1.0, dtype=jnp.float64)  # seed
c_adj = jnp.asarray(0.0, dtype=jnp.float64)
b_adj = jnp.asarray(0.0, dtype=jnp.float64)
a_adj = jnp.asarray(0.0, dtype=jnp.float64)

c_adj += 4.0 * U_adj  # node 4: U = 4c, dU/dc = 4
b_adj += c_adj  # node 3: c = b - a, dc/db = 1
a_adj -= c_adj  # node 3:            dc/da = -1
a_adj += 2.0 * a * b_adj  # node 2: b = a^2, db/da = 2a
r_adj = a_adj * (-6.0 * r**-7)  # node 1: a = r^-6, da/dr = -6 r^-7

# ------------------------------------------- JAX's own gradient at r = 1.3


def lj_pair_energy(r):
    a = r**-6
    b = a**2
    c = b - a
    return 4.0 * c


jax_grad = jax.grad(lj_pair_energy)(r)

# closed form dU/dr = 24 r^-7 - 48 r^-13, as an independent check
analytic = 24.0 * r**-7 - 48.0 * r**-13

# ----------------------------------------------------------- verification
# the three evaluations order the arithmetic differently, so they agree to
# float64 round-off (~1e-15 relative), not bit-for-bit
forward = float(U_tan)
reverse = float(r_adj)
grad_ = float(jax_grad)
analytic_ = float(analytic)
assert math.isclose(forward, reverse, rel_tol=1e-12), (forward, reverse)
assert math.isclose(forward, grad_, rel_tol=1e-12), (forward, grad_)
assert math.isclose(forward, analytic_, rel_tol=1e-12), (forward, analytic_)

# --------------------------------------------------------------- artifacts
out = {
    "r": float(r),
    "energy": float(U),
    "tangents": {
        "r": float(r_tan),
        "a": float(a_tan),
        "b": float(b_tan),
        "c": float(c_tan),
        "U": float(U_tan),
    },
    "adjoints": {
        "r": float(r_adj),
        "a": float(a_adj),
        "b": float(b_adj),
        "c": float(c_adj),
        "U": float(U_adj),
    },
    "jax_grad": grad_,
}

out_path = Path(__file__).resolve().parents[1] / "artifacts" / "ad" / "derivatives.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2) + "\n")

# ----------------------------------------------------------------- report
print(f"r = {float(r)}, U(r) = {float(U)}")
print("\nforward-mode tangents (seed dr = 1):")
for name, tan in out["tangents"].items():
    print(f"  d({name}) = {tan}")
print("\nreverse-mode adjoints (seed dU = 1):")
for name, adj in out["adjoints"].items():
    print(f"  adjoint({name}) = {adj}")
print(f"\njax.grad at r = {grad_}  (closed form {analytic_})")
print(f"\nwrote {out_path}")
