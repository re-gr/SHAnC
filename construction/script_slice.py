import pyvista as pv
import numpy as np
import scipy.spatial as sps

l = 1.0
x = np.linspace(0,l,11)
y = np.linspace(0,l,11)
dS = (x[1] - x[0]) * (y[1]-y[0])

z = np.array([0])
z = np.linspace(0,2*np.pi,41)

X,Y,Z = np.meshgrid(x,y,z)

x_coord = X
y_coord = Y
z_coord = Z
D = 2


plotter = pv.Plotter(window_size=(1200,1200))
plotter.open_gif("density.gif",fps=30,subrectangles=True)

plotter.camera.position =  (-0.8094361996827211, -16.864041357406187, 4.2534831236853865)
plotter.camera.focal_point = (-0.3152552877671506, 0.3673594033096199, 3.3646572316473744)
plotter.camera.roll = 28.53772366213009
# plotter.camera.distance = 5.4641016151376745

L = np.linspace(0,1,100,endpoint=True)

Pos = np.array([X,Y,Z]).transpose((1,2,3,0))

values_init = np.zeros((len(x)-1,len(y)-1,len(z)-1))
for i in range(len(x)-1):
    for j in range(len(y)-1):
        for k in range(len(z)-1):

            CHull =  sps.ConvexHull([Pos[i,j,k],Pos[i,j,k+1],Pos[i,j+1,k],Pos[i+1,j,k],Pos[i,j+1,k+1],Pos[i+1,j,k+1],Pos[i+1,j+1,k],Pos[i+1,j+1,k+1]])
            values_init[i,j,k] = CHull.volume



for l in L:
    rota = l
    Lz = 1
    R = D * rota
    Norm = (Lz**2 + R**2)**(1/2)
    # z_coord = Lz * z
    z_coord = Lz * Z + R * X / Norm
    y_coord = R * np.cos(Z*rota) - np.cos(Z*rota) * Y + Lz * np.sin(Z*rota) / Norm * X
    x_coord = R * np.sin(Z*rota) - np.sin(Z*rota) * Y - Lz * np.cos(Z*rota) / Norm * X


    # print(np.min(z_coord),np.max(z_coord))
    # fil = (z_coord > 4) * (z_coord <= 6)

    Pos = np.array([x_coord,y_coord,z_coord]).transpose((1,2,3,0))



    # print(np.max(x_coord),np.min(x_coord))
    # print(np.max(y_coord),np.min(y_coord))
    # x_coord = X * (1-l*1/2*Y**2)**(1/2)
    # y_coord = Y * (1-l*1/2*X**2)**(1/2)
    # print(np.shape(Z),np.shape(x_coord),np.shape(y_coord))
    # sp = pv.Sphere(radius=0.02,theta_resolution=2,phi_resolution=2)
    # data = pv.PolyData(Pos)
    # pc = data.glyph(scale=False,geom=sp,orient=False)
    # plotter.add_mesh(pc,name="b")

    grid = pv.StructuredGrid(x_coord,y_coord,z_coord)

    values = np.zeros((len(x)-1,len(y)-1,len(z)-1))
    for i in range(len(x)-1):
        for j in range(len(y)-1):
            for k in range(len(z)-1):
                # print(i,j,k)
                # print(np.shape(Pos))
                # print(i,j,k,CHull.volume)
                CHull =  sps.ConvexHull([Pos[i,j,k],Pos[i,j,k+1],Pos[i,j+1,k],Pos[i+1,j,k],Pos[i,j+1,k+1],Pos[i+1,j,k+1],Pos[i+1,j+1,k],Pos[i+1,j+1,k+1]])

                values[i,j,k] = values_init[i,j,k] / CHull.volume
    values = values.transpose((2,1,0))
    grid.cell_data["values"] = values.flatten()


    # plotter.add_mesh(grid,name="b",clim=(1,1.414),cmap="cool")
    # plotter.add_mesh(grid,name="b",cmap="cool")
    plotter.add_mesh(grid,name="b",clim=(0.5,1),cmap="cool",scalar_bar_args={"title":"Density"})
    # plotter.remove_scalar_bar()
    # plotter.update_scalar_bar_title("Density")
    plotter.write_frame()

# print(np.min(values),np.max(values))
# for j in range(10):
#     plotter.write_frame()


# L = np.linspace(1,0,100,endpoint=True)
#
# for l in L:
#     rota = l
#     Lz = 10
#     z = 5
#     R = D * rota
#     Norm = (Lz**2 + D**2)**(1/2)
#     # z_coord = Lz * z
#     y_coord = D * np.cos(z*rota) - np.cos(z*rota) * Y + Lz * np.sin(z*rota) / Norm * X
#     x_coord = D * np.sin(z*rota) - np.sin(z*rota) * Y - Lz * np.cos(z*rota) / Norm * X
#
#
#     grid = pv.StructuredGrid(x_coord,y_coord,Z)
#     values = np.zeros((len(x)-1,len(y)-1))
#     for i in range(len(X)-1):
#         for j in range(len(Y)-1):
#             # values[i,j] = (x_coord[j,i+1,0]-x_coord[j,i,0]) * (y_coord[j+1,i,0]-y_coord[j,i,0])/dS
#             b1 = ((x_coord[j,i+1,0] - x_coord[j,i,0])**2 + (y_coord[j,i+1,0] - y_coord[j,i,0])**2)**(1/2)
#             h1 = ((x_coord[j+1,i,0] - x_coord[j,i,0])**2 + (y_coord[j+1,i,0] - y_coord[j,i,0])**2)**(1/2)
#             b2 = ((x_coord[j,i+1,0] - x_coord[j+1,i+1,0])**2 + (y_coord[j,i+1,0] - y_coord[j+1,i+1,0])**2)**(1/2)
#             h2 = ((x_coord[j+1,i,0] - x_coord[j+1,i+1,0])**2 + (y_coord[j+1,i,0] - y_coord[j+1,i+1,0])**2)**(1/2)
#
#             values[i,j] = dS/((b1*h1 + b2*h2)/2)
#
#
#     grid.cell_data["values"] = values.flatten()
#
#
#     plotter.add_mesh(grid,name="b",clim=(1,1.414),cmap="cool")
#     # plotter.add_mesh(grid,name="b",pbr=True,roughness=1.0,metallic=0.0,clim=(1,1.414),cmap="cool",scalar_bar_args={"title":"Density"})
#     plotter.remove_scalar_bar()
#     plotter.write_frame()
#
#
# # plotter.show()
# plotter.close()