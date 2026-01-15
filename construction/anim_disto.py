from distorsion import *
import pyvista as pv



def transfo2(Pos,slide_z=0,D=0,rota=0,enlarge=10,enlarge_z=0.700758,do_periodic=True,mean=None):

    if type(mean) is type(None):
        mean = np.mean(np.mean(np.mean(Pos,axis=0),axis=0),axis=0)
        mean[2] = np.min(Pos[:,:,:,2])

    Pos = Pos - mean

    Lz = np.max(Pos[:,:,:,2]) / 2/np.pi
    # Lz = np.max(Pos[:,:,:,2])
    # print(Lz)
    x,y,z = Pos.transpose((3,0,1,2))
    z = z/Lz

    Lx = np.max(x)
    lx = np.min(x)
    Ly = np.max(y)
    ly = np.min(y)
    LX = Lx-lx
    LY = Ly-ly
    x = (x-lx - LX/2) / LX * 2
    y = (y-ly - LY/2) / LY * 2



    x_coord = x * (1-1/2*y**2)**(1/2)
    y_coord = y * (1-1/2*x**2)**(1/2)


    x = x_coord * Lx
    y = y_coord * Ly



    # z_coord = (Lz**2-rota**2*np.pi**2*D**2)**(1/2) *(z)/Lz
    # x_coord = (x) * np.cos(2*np.pi*rota*(z/Lz)) - (y+D) * np.sin(2*np.pi*rota*(z/Lz))
    # y_coord = (x) * np.sin(2*np.pi*rota*(z/Lz)) + (y+D) * np.cos(2*np.pi*rota*(z/Lz))-D

    R = D * rota
    Norm = (Lz**2 + R**2)**(1/2)
    z_coord = Lz * z + R * x / Norm

    y_coord = R * np.cos(z*rota) - np.cos(z*rota) * y + Lz * np.sin(z*rota) / Norm * x
    x_coord = R * np.sin(z*rota) - np.sin(z*rota) * y - Lz * np.cos(z*rota) / Norm * x

    # y_coord = R * np.cos(z) - np.cos(z) * y +  np.sin(z) * x
    # x_coord = R * np.sin(z) - np.sin(z) * y -  np.cos(z) * x

    # z_coord = (z_coord >=0.5*Lz) * z_coord +  (z_coord < 0.5*Lz) * (z_coord + Lz)

    #This part is used in order not to slide the inside too much
    if slide_z ==0:
        #slide automatic
        slide_z = np.min(z_coord)
    z_coord = z_coord - slide_z

    if do_periodic:
        z_coord = (z_coord <= Lz*2*np.pi) * z_coord +  (z_coord > Lz*2*np.pi) * (z_coord - Lz * 2 * np.pi)
    # print(np.min(z_coord))

    # print(Lz,np.max(z_coord))
    # print(np.shape(x_coord),np.shape(y_coord),np.shape(z_coord))
    Pos_transfo = np.array([x_coord + mean[0],y_coord + mean[1],z_coord])
    # print(np.shape(Pos_transfo))
    return Pos_transfo,slide_z,mean


if __name__ == "__main__":

    D = 40
    pitch = 300
    width = 80
    thickness = 40
    int_thick = 0
    # pitch = 80
    # width = 30
    # thickness = 20
    # int_thick = 0

    plotter = pv.Plotter(window_size=(800,1000),off_screen=False)

    # plotter = pv.Plotter(window_size=(800,1200),off_screen=True)
    plotter.open_gif("ff.gif",fps=30,loop=0,subrectangles=True)

    Rota = np.linspace(0,1.0,60,endpoint=True)

    plotter.camera.position = (295.6231897664369, 746.0477923773544, 152.16749019607207)
    plotter.camera.focal_point = (44.89679345803219, 21.573329011011676, 179.7240602502272)
    plotter.camera.roll = -82.89593545907061
    plotter.camera.distance = 767.1286322572582
    #
    # nx,ny,nz = 40,20,150
    # x_i = np.linspace(0,80,nx,endpoint=True)
    # y_i = np.linspace(0,40,ny,endpoint=True)
    # z_i = np.linspace(0,300,nz,endpoint=True)
    nx,ny,nz = 10,5,75
    x_i = np.linspace(0,80,nx,endpoint=True)
    y_i = np.linspace(0,40,ny,endpoint=True)
    z_i = np.linspace(0,300,nz,endpoint=True)
    # z_i = np.linspace(0,150,nz,endpoint=True)

    X,Y,Z = np.meshgrid(x_i,y_i,z_i)


    Pos = np.array([X,Y,Z]).transpose((1,2,3,0))
    ##AA
    Pos = Pos.reshape((nx*ny*nz,3))

    # x_i = np.linspace(0,80,nx,endpoint=True)
    # y_i = np.linspace(0,40,ny,endpoint=True)
    # z_i = np.linspace(150,300,nz,endpoint=True)
    #
    # X,Y,Z = np.meshgrid(x_i,y_i,z_i)
    #
    #
    # Pos2 = np.array([X,Y,Z]).transpose((1,2,3,0))
    # Pos2 = Pos2.reshape((nx*ny*nz,3))

    light = pv.Light((-20,150,0),(0,150,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)



    # plotter.add_axes()


    for rota in Rota:
        # Pos_transfo,Types,Lims_tot,Angles_OH, Pos_transfo_int, Types_int, Lims_tot_int = create_syst(rota,D,pitch,width,thickness,int_thick,do_periodic=False)

        Pos_transfo,slide_z,mean = transfo(Pos,slide_z=0,D=D,rota=rota,do_periodic=False,circling=False,do_old_transf=True)
        # Pos_transfo2,slide_z,mean = transfo(Pos2,slide_z=-150,D=D,rota=rota,do_periodic=False,circling=False,do_old_transf=True)
        # x,y,z = Pos_transfo


        # grid = pv.PolyData(x,y,z)
        data = pv.PolyData(Pos_transfo)
        sp = pv.Sphere(radius=1.5)
        pc = data.glyph(scale=False,geom=sp,orient=False)
        # grid2 = pv.PolyData(Pos_transfo2)
        plotter.add_mesh(pc,name="a",opacity=1.0,pbr=False,roughness=0.6,metallic=0.8,color=[56,0,180])
        # plotter.add_mesh(grid2,name="b",opacity=1.0,pbr=True,roughness=0.6,metallic=0.8,color=[180,0,0],render_points_as_spheres=True)
        # grid = pv.StructuredGrid(x,y,z)
        # plotter.add_mesh(grid,name="a",opacity=1.0,pbr=True,roughness=0.6,metallic=0.8,color=[56,0,180])

        if False:
            for z_l in range(10):

                Filter = ((Pos[:,:,:,2]==z_i[z_l*15+5])) + ((Pos[:,:,:,2]==z_i[z_l*15+6]))
                x2,y2,z2 = x[Filter],y[Filter],z[Filter]
                x2 = x2.reshape((ny,nx,2))
                y2 = y2.reshape((ny,nx,2))
                z2 = z2.reshape((ny,nx,2))

                c_x, c_y,c_z = np.mean(x2), np.mean(y2), np.mean(z2)

                Vec = np.array([x2,y2,z2]).transpose((1,2,3,0))
                Vec = Vec - np.array([c_x,c_y,c_z])

                x2 = x2 + 0.01 * Vec[:,:,:,0]
                y2 = y2 + 0.01 * Vec[:,:,:,1]
                z2 = z2 + 0.01 * Vec[:,:,:,2]



                grid_2 = pv.StructuredGrid(x2,y2,z2)

                plotter.add_mesh(grid_2,name="b{}".format(z_l),color="white",pbr=True,roughness=0.6,metallic=1.0,diffuse=True)


        plotter.camera.position = (295.6231897664369, 746.0477923773544, 152.16749019607207)
        plotter.camera.focal_point = (44.89679345803219, 21.573329011011676, 179.7240602502272)
        plotter.camera.roll = -82.89593545907061
        plotter.camera.distance = 767.1286322572582


        plotter.write_frame()
    plotter.write_frame()
    plotter.write_frame()
    plotter.write_frame()
    plotter.write_frame()
    plotter.write_frame()


    plotter.close()
