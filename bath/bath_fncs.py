from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh
from dolfinx.io import VTXWriter
import numpy as np
import ufl

# распаковка чек
mesh = gmsh.read_from_msh("bath.msh", MPI.COMM_WORLD, gdim=3)
bath_mesh = mesh.mesh
bath_cell = mesh.cell_tags
bath_facet = mesh.facet_tags
physical_groups = mesh.physical_groups

hot_surface = physical_groups["hot_surface"].tag
cooling_surface = physical_groups["cooling_surface"].tag
bath = physical_groups["bath"].tag

V = fem.functionspace(bath_mesh, ("Lagrange", 1))

# данные о ванне
T_hot = fem.Constant(bath_mesh, PETSc.ScalarType(80.0))
k_hot = fem.Constant(bath_mesh, PETSc.ScalarType(1000.0))
T_cool = fem.Constant(bath_mesh, PETSc.ScalarType(20.0))
k_cool = fem.Constant(bath_mesh, PETSc.ScalarType(10.0))
# чугун)))
rho = fem.Constant(bath_mesh, PETSc.ScalarType(7000.0))
c = fem.Constant(bath_mesh, PETSc.ScalarType(550.0))
k = fem.Constant(bath_mesh, PETSc.ScalarType(50.0))
# время моделирования
dt = fem.Constant(bath_mesh, PETSc.ScalarType(1))
t_start = 0.0
t_end = 1800.0

# начальные условия
T = fem.Function(V)
T.x.array[:] = 20.0
T.name = "Temp"

# уравнения
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

dx = ufl.dx(domain=bath_mesh)
ds = ufl.Measure("ds", domain=bath_mesh, subdomain_data=bath_facet)

a = (rho * c / dt * u * v * dx
     + k * ufl.dot(ufl.grad(u), ufl.grad(v)) * dx
     + k_hot * u * v * ds(hot_surface)
     + k_cool * u * v * ds(cooling_surface))
L = (rho * c / dt * T * v * dx
     + k_hot * T_hot * v * ds(hot_surface)
     + k_cool * T_cool * v * ds(cooling_surface))

lp = LinearProblem(a, L, petsc_options_prefix="bath_")

# делаем анимацию
with VTXWriter(bath_mesh.comm, "bath.bp", [T], engine="BP4") as vtx:
  vtx.write(t_start)
  while t_start < t_end:
    t_start += float(dt.value)
    T.x.array[:] = lp.solve().x.array[:]
    vtx.write(t_start)