"""gmsh geometries: DFG cylinder benchmark, single cylinder in a large domain, staggered tube bank.

Facet tags: 1 inlet, 2 outlet, 3 walls (channel walls), 4 symmetry, 5 cylinder / tubes. Tube-bank tubes get
tags 10 + row so that each row's heat transfer can be measured.
"""
import gmsh
import numpy as np
from dolfinx.io import gmsh as gio
from mpi4py import MPI

INLET, OUTLET, WALL, SYM, CYL = 1, 2, 3, 4, 5


def _finish(name, h_min, h_max, dist_field_curves, dist_min, dist_max):
    m = gmsh.model
    f = m.mesh.field
    d = f.add('Distance')
    f.setNumbers(d, 'CurvesList', dist_field_curves)
    f.setNumber(d, 'Sampling', 400)
    th = f.add('Threshold')
    f.setNumber(th, 'InField', d)
    f.setNumber(th, 'SizeMin', h_min)
    f.setNumber(th, 'SizeMax', h_max)
    f.setNumber(th, 'DistMin', dist_min)
    f.setNumber(th, 'DistMax', dist_max)
    f.setAsBackgroundMesh(th)
    gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
    gmsh.option.setNumber('Mesh.Algorithm', 6)
    m.mesh.generate(2)
    m.mesh.optimize('Netgen')
    data = gio.model_to_mesh(m, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    return data.mesh, data.facet_tags


def _start(name):
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    gmsh.model.add(name)


def dfg(h_cyl=0.004, h_far=0.03):
    """Schaefer & Turek (1996) 2D-1/2D-2 channel: 2.2 x 0.41, cylinder D = 0.1 at (0.2, 0.2)."""
    _start('dfg')
    occ = gmsh.model.occ
    rect = occ.addRectangle(0, 0, 0, 2.2, 0.41)
    disk = occ.addDisk(0.2, 0.2, 0, 0.05, 0.05)
    fluid, _ = occ.cut([(2, rect)], [(2, disk)])
    occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [fluid[0][1]], 1)
    inl, out, wal, cyl = [], [], [], []
    for dim, tag in gmsh.model.getBoundary(fluid, oriented=False):
        c = occ.getCenterOfMass(dim, tag)
        if np.isclose(c[0], 0):
            inl.append(tag)
        elif np.isclose(c[0], 2.2):
            out.append(tag)
        elif np.isclose(c[1], 0) or np.isclose(c[1], 0.41):
            wal.append(tag)
        else:
            cyl.append(tag)
    for tags, mk in [(inl, INLET), (out, OUTLET), (wal, WALL), (cyl, CYL)]:
        gmsh.model.addPhysicalGroup(1, tags, mk)
    return _finish('dfg', h_cyl, h_far, cyl, 0.02, 0.4)


def cylinder(h_cyl=0.02, h_far=1.5, up=20.0, down=40.0, half=20.0):
    """Single cylinder (D = 1) in uniform flow; slip (symmetry) side boundaries far away."""
    _start('cyl')
    occ = gmsh.model.occ
    rect = occ.addRectangle(-up, -half, 0, up + down, 2 * half)
    disk = occ.addDisk(0, 0, 0, 0.5, 0.5)
    fluid, _ = occ.cut([(2, rect)], [(2, disk)])
    occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [fluid[0][1]], 1)
    groups = {INLET: [], OUTLET: [], SYM: [], CYL: []}
    for dim, tag in gmsh.model.getBoundary(fluid, oriented=False):
        c = occ.getCenterOfMass(dim, tag)
        if np.isclose(c[0], -up):
            groups[INLET].append(tag)
        elif np.isclose(c[0], down):
            groups[OUTLET].append(tag)
        elif np.isclose(abs(c[1]), half):
            groups[SYM].append(tag)
        else:
            groups[CYL].append(tag)
    for mk, tags in groups.items():
        gmsh.model.addPhysicalGroup(1, tags, mk)
    return _finish('cyl', h_cyl, h_far, groups[CYL], 0.5, 15.0)


def tube_bank(rows=8, ST=2.0, SL=2.0, D=1.0, up=3.0, down=8.0, h_tube=0.03, h_far=0.2):
    """Staggered bank, half transverse pitch y in [0, ST/2] with symmetry on both sides.

    Row k sits at x = up + k SL; even rows are centred on y = 0, odd rows on y = ST/2, so every row appears
    as a half tube on one of the symmetry lines (exactly the staggered array by symmetry).
    """
    _start('bank')
    occ = gmsh.model.occ
    H = ST / 2
    L = up + (rows - 1) * SL + down
    rect = occ.addRectangle(0, 0, 0, L, H)
    disks = []
    centres = []
    for k in range(rows):
        xc = up + k * SL
        yc = 0.0 if k % 2 == 0 else H
        disks.append((2, occ.addDisk(xc, yc, 0, D / 2, D / 2)))
        centres.append((xc, yc))
    fluid, _ = occ.cut([(2, rect)], disks)
    occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [fluid[0][1]], 1)
    groups = {INLET: [], OUTLET: [], SYM: []}
    row_tags = {k: [] for k in range(rows)}
    for dim, tag in gmsh.model.getBoundary(fluid, oriented=False):
        c = occ.getCenterOfMass(dim, tag)
        if np.isclose(c[0], 0):
            groups[INLET].append(tag)
        elif np.isclose(c[0], L):
            groups[OUTLET].append(tag)
        elif (np.isclose(c[1], 0) or np.isclose(c[1], H)) and occ.getMass(dim, tag) > 0 and \
                not any(np.hypot(c[0] - xc, c[1] - yc) < D / 2 - 1e-9 for xc, yc in centres):
            groups[SYM].append(tag)
        else:
            k = int(np.argmin([np.hypot(c[0] - xc, c[1] - yc) for xc, yc in centres]))
            row_tags[k].append(tag)
    for mk, tags in groups.items():
        gmsh.model.addPhysicalGroup(1, tags, mk)
    tube_curves = []
    for k, tags in row_tags.items():
        gmsh.model.addPhysicalGroup(1, tags, 10 + k)
        tube_curves += tags
    return _finish('bank', h_tube, h_far, tube_curves, 0.3, 2.0)
