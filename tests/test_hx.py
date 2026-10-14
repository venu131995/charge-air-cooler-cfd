import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_analysis as ra          # noqa: E402
import run_verification as rv      # noqa: E402
from hx import meshes              # noqa: E402


def test_correlations_by_hand():
    # Churchill-Bernstein at Re = 20, Pr = 0.71
    hand = 0.3 + 0.62 * 20 ** 0.5 * 0.71 ** (1 / 3) / (1 + (0.4 / 0.71) ** (2 / 3)) ** 0.25 * (1 + (20 / 282000) ** 0.625) ** 0.8
    assert ra.churchill_bernstein(20.0) == pytest.approx(hand)
    assert float(ra.zukauskas_staggered(50.0)) == pytest.approx(0.90 * 50 ** 0.4 * 0.71 ** 0.36)
    assert np.isnan(float(ra.zukauskas_staggered(500.0)))      # no fit between 100 and 1000


def test_reduced_model_limits():
    assert ra.reduced_outlet_T([0.0] * 8, 50.0, 8) == pytest.approx(0.0)
    # one row: 1 - exp(-NTU)
    V_max, alpha = 2.0, 2.0 / (50.0 * 0.71)
    ntu = 5.0 * alpha * (math.pi / 2) / 1.0
    assert ra.reduced_outlet_T([5.0], 50.0, 1) == pytest.approx(1 - math.exp(-ntu))


def test_kovasznay_and_heat_mms_coarse():
    import ufl
    from dolfinx import mesh
    from mpi4py import MPI
    from hx.solver import Problem, scalar
    Re = 40.0
    lam = Re / 2 - math.sqrt(Re ** 2 / 4 + 4 * math.pi ** 2)
    u_ex = lambda x: np.vstack([1 - np.exp(lam * x[0]) * np.cos(2 * np.pi * x[1]),
                                lam / (2 * np.pi) * np.exp(lam * x[0]) * np.sin(2 * np.pi * x[1])])
    msh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-0.5, -0.5]), np.array([1.0, 1.5])], [16, 21],
                                mesh.CellType.triangle)
    pr = Problem(msh, rv.boundary_tags(msh), {1: 'dirichlet'}, 1 / Re, u_ex,
                 pressure_point=(-0.5, -0.5, 0.5 * (1 - math.exp(2 * lam * -0.5))))
    info = pr.solve_flow(steps=(2, 1))
    assert info['residual_inf'] < 1e-12
    x = ufl.SpatialCoordinate(msh)
    ue = ufl.as_vector([1 - ufl.exp(lam * x[0]) * ufl.cos(2 * ufl.pi * x[1]),
                        lam / (2 * ufl.pi) * ufl.exp(lam * x[0]) * ufl.sin(2 * ufl.pi * x[1])])
    eu = math.sqrt(scalar(ufl.inner(pr.w.sub(0) - ue, pr.w.sub(0) - ue) * ufl.dx(metadata={'quadrature_degree': 8}), msh.comm))
    assert eu < 3e-3
    assert rv.heat_mms(pr, msh)['L2_T'] < 2e-3


def test_small_bank_energy_balance():
    r = ra.bank_case(50.0, rows=2, h_tube=0.06, h_far=0.3)
    assert r['energy_balance_rel'] < 0.02
    assert 0.0 < r['T_out'] < 1.0
    assert r['Nu_row1'] > 0


def test_tube_bank_mesh_tags():
    msh, ft = meshes.tube_bank(rows=3, h_tube=0.08, h_far=0.3)
    tags = set(np.unique(ft.values))
    assert {meshes.INLET, meshes.OUTLET, meshes.SYM, 10, 11, 12} <= tags
