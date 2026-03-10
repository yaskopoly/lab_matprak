import gmsh
import sys

gmsh.initialize()
gmsh.model.add("torus")

r_in = 6.0
r_out = 7.0
R = 10.0
size = (r_out - r_in) / 4

tor_in = gmsh.model.occ.addTorus(0, 0, 0, R, r_in)
tor_out = gmsh.model.occ.addTorus(0, 0, 0, R, r_out)

gmsh.model.occ.cut([(3, tor_out)], [(3, tor_in)], removeObject=True, removeTool=True)

gmsh.model.occ.synchronize()

gmsh.option.setNumber("Mesh.MeshSizeMin", size)
gmsh.option.setNumber("Mesh.MeshSizeMax", size)

gmsh.model.mesh.generate(3)
gmsh.write("torus.msh")

if "-nopopup" not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()