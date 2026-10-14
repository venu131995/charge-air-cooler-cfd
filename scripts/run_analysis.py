"""CFD of cylinder and tube-bank heat transfer, validation against experimental correlations, reduced cooler model.

Writes results/cylinder.csv, results/tube_bank.csv, results/summary.json and figures/fig2-fig5.
Run with the FEniCSx environment (python from ~/micromamba/envs/fx).
"""
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hx import meshes                  # noqa: E402
from hx.solver import Problem          # noqa: E402

RES, FIG = ROOT / 'results', ROOT / 'figures'
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi': 130, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
PR = 0.71


def churchill_bernstein(Re, Pr=PR):
    """Churchill & Bernstein (1977), single cylinder in cross-flow, fitted to experiments for Re Pr > 0.2."""
    return 0.3 + 0.62 * Re ** 0.5 * Pr ** (1 / 3) / (1 + (0.4 / Pr) ** (2 / 3)) ** 0.25 * (1 + (Re / 282000) ** 0.625) ** 0.8


def zukauskas_staggered(Re, Pr=PR, ST=2.0, SL=2.0):
    """Zukauskas (1972), staggered banks of >= 20 rows, constant properties (Pr/Pr_s = 1), from Incropera Table 7.5."""
    Re = np.asarray(Re, float)
    C = np.where(Re <= 100, 0.90, np.where(Re < 1000, np.nan, 0.35 * (ST / SL) ** 0.2))
    m = np.where(Re <= 100, 0.40, 0.60)
    return C * Re ** m * Pr ** 0.36


# ================================================================ single cylinder
def cylinder_case(Re, h_cyl=0.02, h_far=1.5, up=20.0, down=40.0, half=20.0):
    msh, ft = meshes.cylinder(h_cyl=h_cyl, h_far=h_far, up=up, down=down, half=half)
    pr = Problem(msh, ft, {1: 'inlet', 2: 'outlet', 4: 'symmetry', 5: 'wall'}, 1.0 / Re,
                 lambda x: np.vstack([np.ones(x.shape[1]), np.zeros(x.shape[1])]), Pr=PR, wall_heated={5})
    steps = tuple(m for m in (16, 8, 4, 2, 1) if 1.0 / Re * m <= 2.0) or (1,)
    info = pr.solve_flow(steps=steps)
    pr.solve_heat()
    alpha = 1.0 / Re / PR
    q_wall = pr.wall_heat([5])                                    # = h * pi * D * dT / (rho cp)
    q_out = sum(pr.enthalpy_flow(t) + pr.diffusive_flow(t) for t in (1, 2, 4))
    Nu = q_wall / (math.pi * alpha)                               # D = 1, dT = 1
    F = pr.force_volume(5)
    return dict(Re=Re, Nu=Nu, Nu_CB=churchill_bernstein(Re), Cd=2 * F[0], cells=int(msh.topology.index_map(2).size_local),
                energy_balance_rel=abs(q_wall - q_out) / abs(q_wall), newton=info['newton_iterations'],
                flow_residual=info['residual_inf'], h_cyl=h_cyl, domain=f'{up:g}+{down:g} x {2 * half:g}')


# ================================================================ tube bank
def bank_case(Re_max, rows=8, ST=2.0, SL=2.0, h_tube=0.03, h_far=0.2):
    msh, ft = meshes.tube_bank(rows=rows, ST=ST, SL=SL, h_tube=h_tube, h_far=h_far)
    U_in = 1.0
    V_max = U_in * ST / (ST - 1.0)
    nu = V_max * 1.0 / Re_max
    kinds = {1: 'inlet', 2: 'outlet', 4: 'symmetry'}
    kinds.update({10 + k: 'wall' for k in range(rows)})
    pr = Problem(msh, ft, kinds, nu, lambda x: np.vstack([np.full(x.shape[1], U_in), np.zeros(x.shape[1])]), Pr=PR,
                 wall_heated={10 + k for k in range(rows)})
    steps = tuple(m for m in (16, 8, 4, 2, 1) if nu * m <= 1.0) or (1,)
    info = pr.solve_flow(steps=steps)
    pr.solve_heat()
    alpha = nu / PR
    q_rows = np.array([pr.wall_heat([10 + k]) for k in range(rows)])
    q = q_rows.sum()
    q_bc = sum(pr.enthalpy_flow(t) + pr.diffusive_flow(t) for t in (1, 2, 4))
    T_out = pr.bulk_temperature(2)
    A_row = math.pi * 1.0 / 2                                     # one half tube per row in the half-pitch strip
    dT_lm = (1.0 - (1.0 - T_out)) / math.log(1.0 / (1.0 - T_out))
    Nu_bank = q / (rows * A_row * dT_lm) / alpha
    # per-row Nu with the local log-mean temperature difference (bulk temperatures between rows)
    flow = U_in * ST / 2
    Tb = np.concatenate([[0.0], np.cumsum(q_rows) / flow])        # bulk temperature after each row (energy balance)
    lm = ((1 - Tb[:-1]) - (1 - Tb[1:])) / np.log((1 - Tb[:-1]) / (1 - Tb[1:]))
    Nu_row = q_rows / (A_row * lm) / alpha
    # developed rows: from row 2 on, while the bulk temperature still differs from the wall by > 5 %
    # (deeper rows carry too little heat to measure: their Nu is set by round-off in the energy balance)
    theta = 1 - Tb[1:]
    dev = [k for k in range(1, rows) if theta[k] > 0.05]
    Nu_dev = float(np.mean(Nu_row[dev])) if len(dev) >= 2 else float('nan')
    # pressure drop across the rows (averaged over the strip height) per row
    H = ST / 2
    up = 3.0
    ys = np.linspace(0.02, H - 0.02, 21)
    x1, x2 = up - 1.0, up + (rows - 1) * SL + 1.0
    p = pr.w.sub(1).collapse()
    p1 = np.nanmean(pr.point_values(p, np.column_stack([np.full_like(ys, x1), ys])))
    p2 = np.nanmean(pr.point_values(p, np.column_stack([np.full_like(ys, x2), ys])))
    Eu = (p1 - p2) / (0.5 * V_max ** 2) / rows
    return dict(Re_max=Re_max, Nu_bank=Nu_bank, Nu_developed=Nu_dev, developed_rows=[k + 1 for k in dev],
                Nu_row1=float(Nu_row[0]), Nu_zukauskas=float(zukauskas_staggered(Re_max)),
                Nu_single_cylinder_CB=float(churchill_bernstein(Re_max)),
                Nu_rows=Nu_row.tolist(), q_rows=q_rows.tolist(), T_out=T_out, Eu_per_row=float(Eu),
                energy_balance_rel=abs(q - q_bc) / abs(q),
                outlet_T_from_balance=float(Tb[-1]), cells=int(msh.topology.index_map(2).size_local),
                newton=info['newton_iterations'], flow_residual=info['residual_inf'], h_tube=h_tube, rows=rows)


# ================================================================ reduced model (row by row, isothermal tubes)
def reduced_outlet_T(Nu_row, Re_max, rows, ST=2.0, Pr=PR):
    """Row-by-row model: each row is an isothermal surface with NTU_row = Nu alpha A_row / (flow).

    Nondimensional with D = 1, U_in = 1, dT = 1: alpha = V_max / (Re Pr), flow (per strip) = U_in ST/2.
    Outlet bulk temperature after `rows` rows: 1 - prod(exp(-NTU_k)).
    """
    V_max = ST / (ST - 1.0)
    alpha = V_max / (Re_max * Pr)
    A_row = math.pi / 2
    flow = ST / 2
    ntu = np.asarray(Nu_row)[:rows] * alpha * A_row / flow
    return 1.0 - np.prod(np.exp(-ntu))


def main():
    t0 = time.time()
    cyl = []
    for Re in (1.0, 5.0, 10.0, 20.0, 40.0):
        r = cylinder_case(Re)
        cyl.append(r)
        print('   cylinder', {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    # mesh and domain sensitivity at the two ends of the range
    sens = [cylinder_case(40.0, h_cyl=0.01, h_far=1.0), cylinder_case(1.0, up=40.0, down=80.0, half=40.0, h_far=3.0)]
    for r in sens:
        print('   cylinder sensitivity', {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    cdf = pd.DataFrame(cyl)
    cdf.to_csv(RES / 'cylinder.csv', index=False, float_format='%.6g')

    bank = []
    for Re in (10.0, 25.0, 50.0, 100.0, 150.0):
        r = bank_case(Re)
        bank.append(r)
        print('   bank', {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k not in ('Nu_rows', 'q_rows')},
              flush=True)
    bank_fine = bank_case(100.0, h_tube=0.015, h_far=0.1)
    print('   bank fine mesh Re 100', round(bank_fine['Nu_bank'], 4), flush=True)
    bdf = pd.DataFrame(bank)
    bdf.to_csv(RES / 'tube_bank.csv', index=False, float_format='%.6g')

    # reduced model: calibrate per-row Nu from the CFD at each Re, then check against the CFD outlet temperature
    red = []
    for r in bank:
        T_red = reduced_outlet_T(r['Nu_rows'], r['Re_max'], r['rows'])
        # simplest version: one Nu for every row (the bank average) -> error of the 'single Nu' 0-D model
        nu_dev = r['Nu_developed'] if np.isfinite(r['Nu_developed']) else r['Nu_bank']
        T_avg = reduced_outlet_T([r['Nu_row1']] + [nu_dev] * (r['rows'] - 1), r['Re_max'], r['rows'])
        T_zuk = reduced_outlet_T([r['Nu_zukauskas']] * r['rows'], r['Re_max'], r['rows'])
        red.append(dict(Re_max=r['Re_max'], T_out_cfd=r['T_out'], T_out_row_model=T_red, T_out_bank_avg_model=T_avg,
                        T_out_zukauskas_model=T_zuk))
    rdf = pd.DataFrame(red)
    print(rdf.round(5).to_string(), flush=True)

    # power-law fit of the CFD bank Nu, compared with Zukauskas' exponent
    ok = np.isfinite(bdf.Nu_developed)
    fit = np.polyfit(np.log(bdf.Re_max[ok]), np.log(bdf.Nu_developed[ok]), 1)

    # ---------------------------------------------------------------- figures
    fig, ax = plt.subplots(figsize=(5, 3.5))
    Re_l = np.logspace(-0.2, 1.8, 50)
    ax.loglog(Re_l, churchill_bernstein(Re_l), 'k-', lw=1, label='Churchill-Bernstein (experiments)')
    ax.loglog(cdf.Re, cdf.Nu, 'o', color='C3', label='CFD (this work)')
    ax.set(xlabel='Re', ylabel='mean Nusselt number', title='single cylinder, Pr = 0.71')
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / 'fig2_cylinder_nusselt.png')
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    Re_b = np.logspace(1, 2, 30)
    ax[0].loglog(Re_b, zukauskas_staggered(Re_b), 'k-', lw=1, label='Zukauskas (experiments, >= 20 rows)')
    ax[0].loglog(Re_b, churchill_bernstein(Re_b), 'k:', lw=1, label='isolated cylinder (Churchill-Bernstein)')
    ax[0].loglog(bdf.Re_max, bdf.Nu_developed, 'o', color='C0', label='CFD, developed rows')
    ax[0].loglog(bdf.Re_max, bdf.Nu_row1, 's', color='C1', ms=4, label='CFD, first row')
    ax[0].set(xlabel='Re (max velocity, D)', ylabel='Nusselt number', title='staggered bank, ST = SL = 2D')
    ax[0].legend(frameon=False, fontsize=7)
    for r in bank:
        ax[1].plot(range(1, r['rows'] + 1), r['Nu_rows'], 'o-', ms=3, label=f"Re {r['Re_max']:g}")
    ax[1].set(xlabel='row', ylabel='row Nusselt number')
    ax[1].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / 'fig3_tube_bank.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(rdf.Re_max, (rdf.T_out_row_model - rdf.T_out_cfd) / rdf.T_out_cfd * 100, 'o-', label='row-by-row model')
    ax.plot(rdf.Re_max, (rdf.T_out_bank_avg_model - rdf.T_out_cfd) / rdf.T_out_cfd * 100, 's-',
            label='first-row + developed Nu')
    ax.plot(rdf.Re_max, (rdf.T_out_zukauskas_model - rdf.T_out_cfd) / rdf.T_out_cfd * 100, '^-', label='Zukauskas Nu')
    ax.axhline(0, color='grey', lw=0.8)
    ax.set(xlabel='Re (max velocity, D)', ylabel='error in outlet temperature rise (%)', title='reduced models vs CFD')
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / 'fig4_reduced_model.png')
    plt.close(fig)

    summary = dict(cylinder=cyl, cylinder_sensitivity=sens, tube_bank=bank, tube_bank_fine_Re100=bank_fine,
                   reduced_model=red, bank_powerlaw=dict(C=float(math.exp(fit[1])), m=float(fit[0]),
                                                        zukauskas_C_times_Pr036=0.90 * PR ** 0.36, zukauskas_m=0.40),
                   runtime_s=round(time.time() - t0, 1))
    (RES / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    print('runtime', round(time.time() - t0), 's')


if __name__ == '__main__':
    main()
