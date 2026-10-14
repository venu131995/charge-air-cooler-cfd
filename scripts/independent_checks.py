"""Independent checks of the heat-transfer results, with different discretisations of the same physics.

1. Cylinder, Re 20 and 40: temperature in P3 instead of P2, and plain Galerkin (no SUPG) instead of SUPG.
2. Cylinder Nu from the energy method (heat carried out of the domain) instead of the wall flux.
3. Tube bank, Re 100: developed-row Nu with P3 temperature and without SUPG.
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hx import meshes                  # noqa: E402
from hx.solver import Problem          # noqa: E402

PR = 0.71


def cyl(Re, **kw):
    msh, ft = meshes.cylinder()
    pr = Problem(msh, ft, {1: 'inlet', 2: 'outlet', 4: 'symmetry', 5: 'wall'}, 1.0 / Re,
                 lambda x: np.vstack([np.ones(x.shape[1]), np.zeros(x.shape[1])]), Pr=PR, wall_heated={5}, **kw)
    pr.solve_flow(steps=tuple(m for m in (16, 8, 4, 2, 1) if m / Re <= 2.0) or (1,))
    pr.solve_heat()
    alpha = 1.0 / Re / PR
    q_wall = pr.wall_heat([5])
    q_out = sum(pr.enthalpy_flow(t) + pr.diffusive_flow(t) for t in (1, 2, 4))
    return q_wall / (math.pi * alpha), q_out / (math.pi * alpha)


print('1-2. single cylinder Nu: wall flux / energy method')
for Re in (20.0, 40.0):
    for label, kw in [('P2 + SUPG (production)', {}), ('P3 + SUPG', dict(heat_degree=3)),
                      ('P2 Galerkin, no SUPG', dict(supg=False))]:
        nw, ne = cyl(Re, **kw)
        print(f'   Re {Re:g}, {label:24s}: Nu wall flux {nw:.4f}, energy method {ne:.4f}', flush=True)

print('3. tube bank Re 100, developed rows (2-8)')
sys.path.insert(0, str(ROOT / 'scripts'))
import run_analysis as ra              # noqa: E402
for label, kw in [('P2 + SUPG (production)', {}), ('P3 + SUPG', dict(heat_degree=3)), ('P2 Galerkin', dict(supg=False))]:
    orig = ra.Problem
    ra.Problem = lambda *a, _kw=kw, **k: orig(*a, **{**k, **_kw})
    r = ra.bank_case(100.0)
    ra.Problem = orig
    print(f'   {label:24s}: Nu developed {r["Nu_developed"]:.4f}, first row {r["Nu_row1"]:.4f}, '
          f'energy balance {r["energy_balance_rel"]:.2e}', flush=True)
