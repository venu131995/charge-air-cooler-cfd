"""Steady incompressible Navier-Stokes and convective heat transfer in 2-D with DOLFINx.

Flow: Taylor-Hood P2/P1, full Newton iteration with an exact Jacobian (UFL derivative) and MUMPS LU,
      optional continuation in viscosity. rho = 1 (kinematic form).
Heat: P2 temperature, u.grad T - alpha lap T = 0 with SUPG stabilisation (alpha = nu / Pr).
Boundary types are given per facet tag: 'inlet' (prescribed velocity, T = T_in), 'wall' (no slip, T = T_wall or
adiabatic), 'symmetry' (v = 0, adiabatic), 'outlet' (do-nothing, zero diffusive heat flux).
"""
from dataclasses import dataclass, field

import basix.ufl
import dolfinx.fem.petsc as fpetsc
import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

LU = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}


def _linear_solve(A, b, x, prefix):
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOptionsPrefix(prefix)
    opts = PETSc.Options()
    for k, v in LU.items():
        opts[prefix + k] = v
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.solve(b, x)
    ksp.destroy()


def scalar(form, comm):
    return comm.allreduce(fem.assemble_scalar(fem.form(form)), op=MPI.SUM)


class Problem:
    def __init__(self, msh, facet_tags, kinds, nu, u_in, Pr=0.71, T_in=0.0, T_wall=1.0, wall_heated=None,
                 pressure_point=None, heat_degree=2, supg=True):
        """kinds: {tag: 'inlet'|'wall'|'symmetry'|'outlet'|'dirichlet'}; u_in: callable x -> (2, n) inlet velocity.
        wall_heated: set of wall tags held at T_wall (other walls adiabatic). 'dirichlet' = velocity given by u_in
        on that boundary (used for manufactured solutions)."""
        self.msh, self.ft, self.kinds = msh, facet_tags, kinds
        self.comm = msh.comm
        self.nu = fem.Constant(msh, PETSc.ScalarType(nu))
        self.Pr = Pr
        P2 = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(2,))
        P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
        self.W = fem.functionspace(msh, basix.ufl.mixed_element([P2, P1]))
        self.V, _ = self.W.sub(0).collapse()
        self.Q, _ = self.W.sub(1).collapse()
        self.ST = fem.functionspace(msh, ("Lagrange", heat_degree))
        self.supg = supg
        self.w = fem.Function(self.W)
        self.T = fem.Function(self.ST)
        self.ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
        self.n = ufl.FacetNormal(msh)
        fdim = msh.topology.dim - 1
        msh.topology.create_connectivity(fdim, msh.topology.dim)
        self.bcs = []
        uin = fem.Function(self.V)
        uin.interpolate(u_in)
        self.u_in_fn = uin
        zero = fem.Function(self.V)
        V1, _ = self.W.sub(0).sub(1).collapse()
        zero1 = fem.Function(V1)
        for tag, kind in kinds.items():
            facets = facet_tags.find(tag)
            if kind in ('inlet', 'dirichlet'):
                dofs = fem.locate_dofs_topological((self.W.sub(0), self.V), fdim, facets)
                self.bcs.append(fem.dirichletbc(uin, dofs, self.W.sub(0)))
            elif kind == 'wall':
                dofs = fem.locate_dofs_topological((self.W.sub(0), self.V), fdim, facets)
                self.bcs.append(fem.dirichletbc(zero, dofs, self.W.sub(0)))
            elif kind == 'symmetry':
                dofs = fem.locate_dofs_topological((self.W.sub(0).sub(1), V1), fdim, facets)
                self.bcs.append(fem.dirichletbc(zero1, dofs, self.W.sub(0).sub(1)))
        self.has_outlet = any(k == 'outlet' for k in kinds.values())
        if pressure_point is not None:         # pin pressure when no outlet exists (enclosed / MMS problems)
            Qs, qmap = self.W.sub(1).collapse()
            pdof = fem.locate_dofs_geometrical((self.W.sub(1), Qs), lambda x: np.isclose(x[0], pressure_point[0])
                                               & np.isclose(x[1], pressure_point[1]))
            self.p_pin = fem.Function(Qs)
            self.p_pin_value = pressure_point[2]
            self.p_pin.x.array[:] = pressure_point[2]
            self.bcs.append(fem.dirichletbc(self.p_pin, pdof, self.W.sub(1)))
        # heat boundary conditions
        self.T_in, self.T_wall = T_in, T_wall
        self.bcs_T = []
        wall_heated = set() if wall_heated is None else set(wall_heated)
        for tag, kind in kinds.items():
            facets = facet_tags.find(tag)
            dofs = fem.locate_dofs_topological(self.ST, fdim, facets)
            if kind == 'inlet':
                self.bcs_T.append(fem.dirichletbc(PETSc.ScalarType(T_in), dofs, self.ST))
            elif kind == 'wall' and tag in wall_heated:
                self.bcs_T.append(fem.dirichletbc(PETSc.ScalarType(T_wall), dofs, self.ST))
        self.f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))

    # ------------------------------------------------------------------ flow
    def residual(self, w, f=None):
        u, p = ufl.split(w)
        v, q = ufl.TestFunctions(self.W)
        f = self.f if f is None else f
        return (self.nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
                + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
                - p * ufl.div(v) * ufl.dx - q * ufl.div(u) * ufl.dx - ufl.inner(f, v) * ufl.dx)

    def solve_flow(self, nu_target=None, steps=(1,), tol=1e-11, max_it=30, verbose=False, f=None):
        """Newton iteration; 'steps' gives viscosity multipliers for continuation (e.g. (8, 4, 2, 1))."""
        nu_final = float(self.nu.value) if nu_target is None else nu_target
        it_total = 0
        history = []
        for mult in steps:
            self.nu.value = nu_final * mult
            F = self.residual(self.w, f)
            J = ufl.derivative(F, self.w)
            Ff, Jf = fem.form(F), fem.form(J)
            dw = fem.Function(self.W)
            for it in range(max_it):
                A = fpetsc.assemble_matrix(Jf, bcs=self.bcs)
                A.assemble()
                b = fpetsc.assemble_vector(Ff)
                # Newton on the correction: bcs applied to w - g
                fpetsc.apply_lifting(b, [Jf], bcs=[self.bcs], x0=[self.w.x.petsc_vec], alpha=-1.0)
                b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                for bc in self.bcs:
                    bc.set(b.array_w, x0=self.w.x.array, alpha=-1.0)
                b.scale(-1.0)
                rn = b.norm()
                _linear_solve(A, b, dw.x.petsc_vec, f'ns{id(self)}_')
                dw.x.scatter_forward()
                self.w.x.array[:] += dw.x.array
                self.w.x.scatter_forward()
                dn = dw.x.petsc_vec.norm() / max(self.w.x.petsc_vec.norm(), 1e-300)
                history.append((mult, it, rn, dn))
                A.destroy()
                b.destroy()
                it_total += 1
                if verbose and self.comm.rank == 0:
                    print(f'      nu x{mult}: Newton {it} |F| {rn:.3e} |du|/|u| {dn:.3e}', flush=True)
                if dn < tol:
                    break
        self.nu.value = nu_final
        # final residual of the discrete equations (owned dofs not constrained by Dirichlet conditions)
        b = fpetsc.assemble_vector(fem.form(self.residual(self.w, f)))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        bc_dofs = np.concatenate([bc.dof_indices()[0] for bc in self.bcs]) if self.bcs else np.array([], int)
        arr = b.array_r.copy()
        arr[bc_dofs[bc_dofs < len(arr)]] = 0.0
        self.flow_residual = float(np.abs(arr).max())
        b.destroy()
        return dict(newton_iterations=it_total, history=history, residual_inf=self.flow_residual)

    # ------------------------------------------------------------------ heat
    def heat_forms(self, T, s, source=None):
        u = self.w.sub(0)
        alpha = self.nu / self.Pr
        h = ufl.CellDiameter(self.msh)
        unorm = ufl.sqrt(ufl.inner(u, u) + 1e-30)
        pe = unorm * h / (2 * alpha)
        tau = h / (2 * unorm) * (1 / ufl.tanh(pe) - 1 / pe)
        src = 0.0 if source is None else source
        R = ufl.inner(u, ufl.grad(T)) - alpha * ufl.div(ufl.grad(T)) - src
        a = (ufl.inner(u, ufl.grad(T)) * s + alpha * ufl.inner(ufl.grad(T), ufl.grad(s)) - src * s) * ufl.dx \
            + (tau * R * ufl.inner(u, ufl.grad(s)) * ufl.dx if self.supg else 0)
        return a

    def solve_heat(self, source=None, bcs=None):
        Ttr, s = ufl.TrialFunction(self.ST), ufl.TestFunction(self.ST)
        a_full = self.heat_forms(Ttr, s, source)
        a, L = ufl.system(a_full)
        if L is None or (hasattr(L, 'empty') and L.empty()):
            L = fem.Constant(self.msh, PETSc.ScalarType(0.0)) * s * ufl.dx
        bcs = self.bcs_T if bcs is None else bcs
        af, Lf = fem.form(a), fem.form(L)
        A = fpetsc.assemble_matrix(af, bcs=bcs)
        A.assemble()
        b = fpetsc.assemble_vector(Lf)
        fpetsc.apply_lifting(b, [af], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        for bc in bcs:
            bc.set(b.array_w)
        _linear_solve(A, b, self.T.x.petsc_vec, f'heat{id(self)}_')
        self.T.x.scatter_forward()
        A.destroy()
        b.destroy()

    # ------------------------------------------------------------------ post-processing
    def force(self, tag):
        """Force of the fluid on the boundary 'tag' (per unit depth, rho = 1)."""
        u, p = self.w.sub(0), self.w.sub(1)
        sigma = -p * ufl.Identity(2) + self.nu * (ufl.grad(u) + ufl.grad(u).T)
        t = -ufl.dot(sigma, self.n)
        return np.array([scalar(t[i] * self.ds(tag), self.comm) for i in range(2)])

    def force_volume(self, tag):
        """Force on boundary 'tag' from the weak residual tested with a lifting function phi = e_i on 'tag'
        (zero on all other Dirichlet boundaries). Much more accurate than the surface integral (Babuska-Miller)."""
        fdim = self.msh.topology.dim - 1
        others = np.concatenate([self.ft.find(tg) for tg, k in self.kinds.items()
                                 if k in ('inlet', 'wall', 'dirichlet') and tg != tag] or [np.array([], np.int32)])
        out = []
        for i in range(2):
            phi = fem.Function(self.V)
            dofs_t = fem.locate_dofs_topological(self.V, fdim, self.ft.find(tag))
            bs = self.V.dofmap.index_map_bs
            arr = phi.x.array.reshape(-1, bs)
            arr[dofs_t, i] = 1.0
            phi.x.scatter_forward()
            u, p = ufl.split(self.w)
            R = (self.nu * ufl.inner(ufl.grad(u), ufl.grad(phi)) + ufl.inner(ufl.grad(u) * u, phi)
                 - p * ufl.div(phi)) * ufl.dx
            out.append(-scalar(R, self.comm))
        return np.array(out)

    def wall_heat(self, tags):
        """Heat flowing from the given walls into the fluid, per unit depth, in units of alpha * dT."""
        alpha = float(self.nu.value) / self.Pr
        return sum(scalar(alpha * ufl.dot(ufl.grad(self.T), self.n) * self.ds(tg), self.comm) for tg in tags)

    def enthalpy_flow(self, tag):
        """Advected heat leaving through boundary 'tag' (u.n T), per unit depth."""
        u = self.w.sub(0)
        return scalar(self.T * ufl.dot(u, self.n) * self.ds(tag), self.comm)

    def diffusive_flow(self, tag):
        alpha = float(self.nu.value) / self.Pr
        return scalar(-alpha * ufl.dot(ufl.grad(self.T), self.n) * self.ds(tag), self.comm)

    def flow_rate(self, tag):
        return scalar(ufl.dot(self.w.sub(0), self.n) * self.ds(tag), self.comm)

    def bulk_temperature(self, tag):
        u = self.w.sub(0)
        return scalar(self.T * ufl.dot(u, self.n) * self.ds(tag), self.comm) / scalar(ufl.dot(u, self.n) * self.ds(tag), self.comm)

    def length(self, tag):
        return scalar(fem.Constant(self.msh, PETSc.ScalarType(1.0)) * self.ds(tag), self.comm)

    def point_values(self, func, pts):
        """Evaluate a function at points (serial use)."""
        from dolfinx import geometry
        pts3 = np.zeros((len(pts), 3))
        pts3[:, :2] = pts
        tree = geometry.bb_tree(self.msh, self.msh.topology.dim)
        cand = geometry.compute_collisions_points(tree, pts3)
        cells = geometry.compute_colliding_cells(self.msh, cand, pts3)
        out = []
        for i in range(len(pts)):
            c = cells.links(i)
            out.append(func.eval(pts3[i], c[:1]) if len(c) else np.full(func.ufl_shape or (1,), np.nan))
        return np.array(out)
