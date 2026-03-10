import gmsh
import sys
import numpy as np

gmsh.initialize()
gmsh.model.add("new_stl")

gmsh.merge("gear.stl")

angle = 30 * np.pi / 180

gmsh.model.mesh.classifySurfaces(
  angle=angle, 
  boundary=True, 
  forReparametrization=True
)

gmsh.model.mesh.createGeometry()

surfaces = gmsh.model.getEntities(2)
sl = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces])
gmsh.model.geo.addVolume([sl])
gmsh.model.geo.synchronize()

gmsh.model.mesh.generate(3)

gmsh.write("new_stl.msh")

if "-nopopup" not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()