from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import gmsh

import ufl
from dolfinx.fem import (
  Expression,
  Function,
  assemble_scalar,
  dirichletbc,
  form,
  functionspace,
  locate_dofs_geometrical,
)

from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.io import gmsh as gmsh_d

k0 = 4 * np.pi
deg = 1
n_elem = 64
A = 1
r = 2
l = 2 * r / n_elem

gmsh.initialize()
gmsh.model.add("shar")
shar = gmsh.model.occ.addSphere(0, 0, 0, r)
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(3, [shar], 1)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", l)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", l)
gmsh.model.mesh.generate(3)
msh = (gmsh_d.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)).mesh
gmsh.finalize()

V = functionspace(msh, ("Lagrange", deg))

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k0**2 * ufl.inner(u, v) * ufl.dx

theta = np.pi / 4
V_exact = functionspace(
  msh, ("Lagrange", deg + 3)
)
u_exact = Function(V_exact, name="u_exact")
u_exact.interpolate(
  lambda x: A * np.cos(k0 * (np.cos(theta) * x[0] + np.sin(theta) * x[1]))
)
x = ufl.SpatialCoordinate(msh)
n = ufl.FacetNormal(msh)
g = -ufl.dot(n, ufl.grad(u_exact))
L = -ufl.inner(g, v) * ufl.ds

dofs_D = locate_dofs_geometrical(
  V,
  lambda x: np.logical_and(x[0] <= 0, np.isclose(np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2), r))
)

u_bc = Function(V)
u_bc.interpolate(Expression(u_exact, V.element.interpolation_points))
bcs = [dirichletbc(u_bc, dofs_D)]

is_complex_mode = np.issubdtype(PETSc.ScalarType, np.complexfloating)
PETSc.Sys.Print(f"PETSc is configured in complex mode: {is_complex_mode}")

uh = Function(V)
uh.name = "u"
problem = LinearProblem(
  a,
  L,
  bcs=bcs,
  u=uh,
  petsc_options_prefix="demo_helmholtz_",
  petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True},
)
_ = problem.solve()

with XDMFFile(
  MPI.COMM_WORLD, "out_helmholtz/plane_wave.xdmf", "w", encoding=XDMFFile.Encoding.HDF5
) as file:
  file.write_mesh(msh)
  file.write_function(uh)

diff = uh - u_exact
H1_diff = msh.comm.allreduce(
  assemble_scalar(form(ufl.inner(ufl.grad(diff), ufl.grad(diff)) * ufl.dx)), op=MPI.SUM
)
H1_exact = msh.comm.allreduce(
  assemble_scalar(form(ufl.inner(ufl.grad(u_exact), ufl.grad(u_exact)) * ufl.dx)), op=MPI.SUM
)
PETSc.Sys.Print("Relative H1 error of FEM solution:", abs(np.sqrt(H1_diff) / np.sqrt(H1_exact)))

L2_diff = msh.comm.allreduce(assemble_scalar(form(ufl.inner(diff, diff) * ufl.dx)), op=MPI.SUM)
L2_exact = msh.comm.allreduce(
  assemble_scalar(form(ufl.inner(u_exact, u_exact) * ufl.dx)), op=MPI.SUM
)
PETSc.Sys.Print("Relative L2 error of FEM solution:", abs(np.sqrt(L2_diff) / np.sqrt(L2_exact)))