"""Verification of the CFD solver: exact Navier-Stokes solution, manufactured heat solution, DFG benchmark.

Writes results/verification.json and figures/fig1_verification.png. Run with the FEniCSx environment.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hx import meshes                  # noqa: E402
from hx.solver import Problem, scalar  # noqa: E402

OUT = {}


def boundary_tags(msh):
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    facets = mesh.exterior_facet_indices(msh.topology)
    return mesh.meshtags(msh, fdim, facets, np.full(len(facets), 1, np.int32))


# ---------------------------------------------------------------- 1. Kovasznay flow (exact NS solution, Re = 40)
def kovasznay():
    Re = 40.0
    lam = Re / 2 - math.sqrt(Re ** 2 / 4 + 4 * math.pi ** 2)
    u_ex = lambda x: np.vstack([1 - np.exp(lam * x[0]) * np.cos(2 * np.pi * x[1]),
                                lam / (2 * np.pi) * np.exp(lam * x[0]) * np.sin(2 * np.pi * x[1])])
    res = {}
    for n in (8, 16, 32, 64):
        msh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-0.5, -0.5]), np.array([1.0, 1.5])], [n, int(4 * n / 3)],
                                    mesh.CellType.triangle)
        ft = boundary_tags(msh)
        pr = Problem(msh, ft, {1: 'dirichlet'}, 1 / Re, u_ex,
                     pressure_point=(-0.5, -0.5, 0.5 * (1 - math.exp(2 * lam * -0.5))))
        info = pr.solve_flow(steps=(2, 1))
        x = ufl.SpatialCoordinate(msh)
        ue = ufl.as_vector([1 - ufl.exp(lam * x[0]) * ufl.cos(2 * ufl.pi * x[1]),
                            lam / (2 * ufl.pi) * ufl.exp(lam * x[0]) * ufl.sin(2 * ufl.pi * x[1])])
        pe = 0.5 * (1 - ufl.exp(2 * lam * x[0]))
        u, p = pr.w.sub(0), pr.w.sub(1)
        dxq = ufl.dx(metadata={'quadrature_degree': 8})
        eu = math.sqrt(scalar(ufl.inner(u - ue, u - ue) * dxq, msh.comm))
        egu = math.sqrt(scalar(ufl.inner(ufl.grad(u - ue), ufl.grad(u - ue)) * dxq, msh.comm))
        ep = math.sqrt(scalar((p - pe) ** 2 * dxq, msh.comm))
        divu = math.sqrt(scalar(ufl.div(u) ** 2 * dxq, msh.comm))
        res[n] = dict(L2_u=eu, H1_u=egu, L2_p=ep, L2_div_u=divu, newton=info['newton_iterations'],
                      residual_inf=info['residual_inf'])
        # also: verify the heat solver on this flow with a manufactured temperature
        res[n].update(heat_mms(pr, msh))
        print('   kovasznay', n, {k: f'{v:.3e}' for k, v in res[n].items()}, flush=True)
    ns = np.array(list(res))
    order = {k: float(-np.polyfit(np.log(ns[1:]), np.log([res[n][k] for n in ns[1:]]), 1)[0])
             for k in ('L2_u', 'H1_u', 'L2_p', 'L2_T', 'H1_T')}
    return dict(errors={str(k): v for k, v in res.items()}, orders=order)


def heat_mms(pr, msh):
    """T = sin(pi x) cos(pi y) + x^2 with the computed Kovasznay velocity; source chosen to make it exact."""
    x = ufl.SpatialCoordinate(msh)
    Te = ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]) + x[0] ** 2
    u = pr.w.sub(0)
    alpha = float(pr.nu.value) / pr.Pr
    src = ufl.inner(u, ufl.grad(Te)) - alpha * ufl.div(ufl.grad(Te))
    Tb = fem.Function(pr.ST)
    Tb.interpolate(lambda X: np.sin(np.pi * X[0]) * np.cos(np.pi * X[1]) + X[0] ** 2)
    fdim = msh.topology.dim - 1
    dofs = fem.locate_dofs_topological(pr.ST, fdim, mesh.exterior_facet_indices(msh.topology))
    pr.solve_heat(source=src, bcs=[fem.dirichletbc(Tb, dofs)])
    dxq = ufl.dx(metadata={'quadrature_degree': 8})
    eT = math.sqrt(scalar((pr.T - Te) ** 2 * dxq, msh.comm))
    egT = math.sqrt(scalar(ufl.inner(ufl.grad(pr.T - Te), ufl.grad(pr.T - Te)) * dxq, msh.comm))
    return dict(L2_T=eT, H1_T=egT)


# ---------------------------------------------------------------- 2. DFG 2D-1 benchmark (Schaefer & Turek 1996)
REF = dict(cD=5.57953523384, cL=0.010618948146, dp=0.11752016697)   # John & Matthies (2001) reference values


def dfg():
    res = {}
    for h in (0.008, 0.004, 0.002):
        t0 = time.time()
        msh, ft = meshes.dfg(h_cyl=h, h_far=min(0.03, 8 * h))
        Um, H = 0.3, 0.41
        pr = Problem(msh, ft, {1: 'inlet', 2: 'outlet', 3: 'wall', 5: 'wall'}, 1e-3,
                     lambda x: np.vstack([4 * Um * x[1] * (H - x[1]) / H ** 2, 0 * x[0]]))
        info = pr.solve_flow(steps=(4, 2, 1))
        U = 2 * Um / 3
        F = pr.force_volume(5)
        Fs = pr.force(5)
        pv = pr.point_values(pr.w.sub(1).collapse(), np.array([[0.15, 0.2], [0.25, 0.2]]))
        res[str(h)] = dict(cells=int(msh.topology.index_map(2).size_local), cD=2 * F[0] / (U ** 2 * 0.1),
                           cL=2 * F[1] / (U ** 2 * 0.1), cD_surface=2 * Fs[0] / (U ** 2 * 0.1),
                           dp=float(pv[0][0] - pv[1][0]),
                           mass_imbalance=abs(pr.flow_rate(1) + pr.flow_rate(2)) / abs(pr.flow_rate(1)),
                           newton=info['newton_iterations'], residual_inf=info['residual_inf'],
                           runtime_s=time.time() - t0)
        for k in ('cD', 'cL', 'dp'):
            res[str(h)][k + '_rel_err'] = abs(res[str(h)][k] - REF[k]) / REF[k]
        print('   dfg', h, {k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in res[str(h)].items()}, flush=True)
    return dict(reference=REF, runs=res)


def main():
    print('1 Kovasznay + heat MMS', flush=True)
    OUT['kovasznay'] = kovasznay()
    print(OUT['kovasznay']['orders'], flush=True)
    print('2 DFG 2D-1', flush=True)
    OUT['dfg_2d1'] = dfg()
    (ROOT / 'results' / 'verification.json').write_text(json.dumps(OUT, indent=2))


if __name__ == '__main__':
    main()
