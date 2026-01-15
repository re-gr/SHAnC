import pyvista as pv
import numpy as np

l = 1.0
x = np.linspace(-l,l,101)
y = np.linspace(-l,l,101)
dS = (x[1] - x[0]) * (y[1]-y[0])
z = np.array([0])

X,Y,Z = np.meshgrid(x,y,z)

x_coord = X
y_coord = Y

plotter = pv.Plotter(window_size=(1200,1200))
plotter.open_gif("rounding.gif",fps=30,subrectangles=True)

plotter.camera.position =  (-0.01, 0, 5.5)
plotter.camera.focal_point = (0.0, 0.0, 0.0)
plotter.camera.roll = 0
# plotter.camera.distance = 5.4641016151376745

L = np.linspace(0,1,100,endpoint=True)

for j in range(10):
    grid = pv.StructuredGrid(X,Y,Z)
    values = np.zeros((len(x)-1,len(y)-1))
    for i in range(len(X)-1):
        for j in range(len(Y)-1):
            # values[i,j] = (x_coord[j,i+1,0]-x_coord[j,i,0]) * (y_coord[j+1,i,0]-y_coord[j,i,0])/dS
            b1 = ((x_coord[j,i+1,0] - x_coord[j,i,0])**2 + (y_coord[j,i+1,0] - y_coord[j,i,0])**2)**(1/2)
            h1 = ((x_coord[j+1,i,0] - x_coord[j,i,0])**2 + (y_coord[j+1,i,0] - y_coord[j,i,0])**2)**(1/2)
            b2 = ((x_coord[j,i+1,0] - x_coord[j+1,i+1,0])**2 + (y_coord[j,i+1,0] - y_coord[j+1,i+1,0])**2)**(1/2)
            h2 = ((x_coord[j+1,i,0] - x_coord[j+1,i+1,0])**2 + (y_coord[j+1,i,0] - y_coord[j+1,i+1,0])**2)**(1/2)

            values[i,j] = dS/((b1*h1 + b2*h2)/2)
    grid.cell_data["values"] = values.flatten()
    print(np.min(values),np.max(values))


    plotter.add_mesh(grid,name="b",clim=(1,1.414),cmap="cool")
    plotter.remove_scalar_bar()


    plotter.write_frame()



for l in L:

    x_coord = X * (1-l*1/2*Y**2)**(1/2)
    y_coord = Y * (1-l*1/2*X**2)**(1/2)


    grid = pv.StructuredGrid(x_coord,y_coord,Z)
    values = np.zeros((len(x)-1,len(y)-1))
    for i in range(len(X)-1):
        for j in range(len(Y)-1):
            # values[i,j] = (x_coord[j,i+1,0]-x_coord[j,i,0]) * (y_coord[j+1,i,0]-y_coord[j,i,0])/dS
            b1 = ((x_coord[j,i+1,0] - x_coord[j,i,0])**2 + (y_coord[j,i+1,0] - y_coord[j,i,0])**2)**(1/2)
            h1 = ((x_coord[j+1,i,0] - x_coord[j,i,0])**2 + (y_coord[j+1,i,0] - y_coord[j,i,0])**2)**(1/2)
            b2 = ((x_coord[j,i+1,0] - x_coord[j+1,i+1,0])**2 + (y_coord[j,i+1,0] - y_coord[j+1,i+1,0])**2)**(1/2)
            h2 = ((x_coord[j+1,i,0] - x_coord[j+1,i+1,0])**2 + (y_coord[j+1,i,0] - y_coord[j+1,i+1,0])**2)**(1/2)

            values[i,j] = dS/((b1*h1 + b2*h2)/2)

    grid.cell_data["values"] = values.flatten()


    plotter.add_mesh(grid,name="b",clim=(1,1.414),cmap="cool")
    plotter.remove_scalar_bar()
    # plotter.add_mesh(grid,name="b",pbr=True,roughness=1.0,metallic=0.0,clim=(0.5,1),cmap="cool",scalar_bar_args={"title":"Density"})
    # plotter.update_scalar_bar_title("Density")
    plotter.write_frame()

print(np.min(values),np.max(values))
for j in range(10):
    plotter.write_frame()


L = np.linspace(1,0,100,endpoint=True)

for l in L:

    x_coord = X * (1-l*1/2*Y**2)**(1/2)
    y_coord = Y * (1-l*1/2*X**2)**(1/2)


    grid = pv.StructuredGrid(x_coord,y_coord,Z)
    values = np.zeros((len(x)-1,len(y)-1))
    for i in range(len(X)-1):
        for j in range(len(Y)-1):
            # values[i,j] = (x_coord[j,i+1,0]-x_coord[j,i,0]) * (y_coord[j+1,i,0]-y_coord[j,i,0])/dS
            b1 = ((x_coord[j,i+1,0] - x_coord[j,i,0])**2 + (y_coord[j,i+1,0] - y_coord[j,i,0])**2)**(1/2)
            h1 = ((x_coord[j+1,i,0] - x_coord[j,i,0])**2 + (y_coord[j+1,i,0] - y_coord[j,i,0])**2)**(1/2)
            b2 = ((x_coord[j,i+1,0] - x_coord[j+1,i+1,0])**2 + (y_coord[j,i+1,0] - y_coord[j+1,i+1,0])**2)**(1/2)
            h2 = ((x_coord[j+1,i,0] - x_coord[j+1,i+1,0])**2 + (y_coord[j+1,i,0] - y_coord[j+1,i+1,0])**2)**(1/2)

            values[i,j] = dS/((b1*h1 + b2*h2)/2)


    grid.cell_data["values"] = values.flatten()


    plotter.add_mesh(grid,name="b",clim=(1,1.414),cmap="cool")
    # plotter.add_mesh(grid,name="b",pbr=True,roughness=1.0,metallic=0.0,clim=(1,1.414),cmap="cool",scalar_bar_args={"title":"Density"})
    plotter.remove_scalar_bar()
    plotter.write_frame()


# plotter.show()
plotter.close()