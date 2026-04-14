import gmsh
import sys
import numpy as np

gmsh.initialize()
gmsh.model.add("bath")

gmsh.model.occ.importShapes("bath.step")
gmsh.model.occ.synchronize()

bath = gmsh.model.getEntities(3)[0][1] # создаем вспомогательный цилиндр ака горячая вода льется на поверхность
hot_surface = gmsh.model.occ.addCylinder(0.57, 0, -0.3, 0, 0, 0.6, 0.09)
gmsh.model.occ.fragment([(3, bath)], [(3, hot_surface)])
gmsh.model.occ.synchronize()

max_m = -1
max_m_t = None
for i, t in gmsh.model.getEntities(3): # по максимальной массе определяем ванну
  m = gmsh.model.occ.getMass(i, t)
  if m > max_m:
    max_m = m
    max_m_t = t

bath = max_m_t
bath_surface = set(t for i, t in gmsh.model.getBoundary([(3, bath)], False, False))

water = set()
for i, t in gmsh.model.getEntities(3): # в качестве воды - все остальное, кроме ванны
  if t != bath:
    water.update([t_hot for j, t_hot in gmsh.model.getBoundary([(3, t)], False, False)])

hot_surface = list(bath_surface.intersection(water)) # поверхность нагрева
cooling_surface = list(bath_surface - set(hot_surface)) # вся поверхность, которая не нагрев - охлаждение

gmsh.model.addPhysicalGroup(2, hot_surface, name="hot_surface")
gmsh.model.addPhysicalGroup(2, cooling_surface, name="cooling_surface")
gmsh.model.addPhysicalGroup(3, [bath], name="bath")

gmsh.model.mesh.generate(3)
gmsh.write("bath.msh")
gmsh.finalize()