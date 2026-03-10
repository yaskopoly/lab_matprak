import numpy as np
import vtk

class CalcMesh:
  def __init__(self, file, size, omega, omega_w, k, step):
    self.size = size
    self.omega = omega
    self.omega_w = omega_w
    self.k = k
    self.step = step
    self.time = 0

    read = vtk.vtkSTLReader()
    read.SetFileName(file)
    read.Update()

    x_mn, x_mx, y_mn, y_mx, z_mn, z_mx = read.GetOutput().GetBounds()
    x = np.linspace(x_mn, x_mx, size)
    y = np.linspace(y_mn, y_mx, size)
    z = np.linspace(z_mn, z_mx, size)

    self.start = np.array(np.meshgrid(x, y, z, indexing="ij"), np.double)
    self.now = self.start.copy()

    self.v = np.zeros((3, size, size, size), np.double)
    self.smth = np.zeros((size, size, size), np.double)
    self.inside = np.zeros((size, size, size), np.uint8)

    self.x_cntr = (x_mn + x_mx) / 2
    self.y_cntr = (y_mn + y_mx) / 2
    self.z_cntr = (z_mn + z_mx) / 2

    points_dataset = vtk.vtkPolyData()
    points = vtk.vtkPoints()

    for i in range(size):
      for j in range(size):
        for l in range(size):
          points.InsertNextPoint(self.start[0][i, j, l], self.start[1][i, j, l],
                                 self.start[2][i, j, l])

    points_dataset.SetPoints(points)

    encl = vtk.vtkSelectEnclosedPoints()
    encl.SetInputData(points_dataset)
    encl.SetSurfaceData(read.GetOutput())
    encl.Update()

    arr_incl_points = encl.GetOutput().GetPointData().GetArray("SelectedPoints")

    idx = 0
    for i in range(size):
      for j in range(size):
        for l in range(size):
          self.inside[i, j, l] = arr_incl_points.GetValue(idx)
          idx += 1

    self.calculate()

  def calculate(self):
    dx = self.now[0] - self.x_cntr
    dy = self.now[1] - self.y_cntr
    r = np.sqrt(dx * dx + dy * dy)
    self.smth = 1.0 + np.sin(self.k * r - self.omega_w * self.time)

    self.v[0] = -self.omega * dy
    self.v[1] = self.omega * dx
    self.v[2] = 0

    incl = self.inside == 1

    self.v[0] *= incl
    self.v[1] *= incl
    self.v[2] *= incl
    self.smth *= incl

  def move(self):
    self.time += self.step

    dx = self.start[0] - self.x_cntr
    dy = self.start[1] - self.y_cntr

    angle = self.omega * self.time
    cos = np.cos(angle)
    sin = np.sin(angle)

    self.now[0] = self.x_cntr + cos * dx - sin * dy
    self.now[1] = self.y_cntr + sin * dx + cos * dy
    self.now[2] = self.start[2]

    self.calculate()

  def snapshot(self, snap_number):
    structuredGrid = vtk.vtkStructuredGrid()
    points = vtk.vtkPoints()

    smth = vtk.vtkDoubleArray()
    smth.SetName("smth")

    vel = vtk.vtkDoubleArray()
    vel.SetNumberOfComponents(3)
    vel.SetName("vel")

    inside = vtk.vtkDoubleArray()
    inside.SetName("inside")

    number = self.size
    for i in range(number):
      for j in range(number):
        for l in range(number):
          points.InsertNextPoint(self.now[0][i, j, l], self.now[1][i, j, l],
                                 self.now[2][i, j, l])
          smth.InsertNextValue(self.smth[i, j, l])
          vel.InsertNextTuple((self.v[0][i, j, l], self.v[1][i, j, l],
                               self.v[2][i, j, l]))
          inside.InsertNextValue(self.inside[i, j, l])

    structuredGrid.SetDimensions(number, number, number)
    structuredGrid.SetPoints(points)

    structuredGrid.GetPointData().AddArray(smth)
    structuredGrid.GetPointData().AddArray(vel)
    structuredGrid.GetPointData().AddArray(inside)

    structuredGrid.GetPointData().SetActiveScalars("smth")
    structuredGrid.GetPointData().SetActiveVectors("vel")

    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetInputDataObject(structuredGrid)
    writer.SetFileName("gear_vtk/gear-step-" + str(snap_number) + ".vts")
    writer.SetDataModeToAscii()
    writer.Write()

size = 70
fps = 35
total_time = 5.0
step = 1.0 / fps

m = CalcMesh("gear.stl", size, 2, 10, 5, step)

for n in range(int(total_time * fps) + 1):
  m.snapshot(n)
  m.move()