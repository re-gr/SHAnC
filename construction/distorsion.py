import numpy as np
# import matplotlib.pyplot as plt
from script_analysis import *

def read_data(file):
    Lims = []
    Atom_types = []
    Atom_pos = []
    for line in open(file):
        lsplit = line.split()
        if len(lsplit) == 4:
            Lims.append([float(lsplit[0]),float(lsplit[1])])

        elif len(lsplit) == 6:
            Atom_types.append(int(lsplit[1]))
            Atom_pos.append([float(lsplit[3]),float(lsplit[4]),float(lsplit[5])])
    Atom_pos = np.array(Atom_pos)
    Atom_pos = Atom_pos - np.array([0,0,np.min(Atom_pos[:,2])])
    Lims[2][1] = Lims[2][1] - Lims[2][0]
    Lims[2][0] = 0
    return np.array(Lims),np.array(Atom_types),Atom_pos




def duplicate(Nx_list,Ny_list,Nz,Lims,Atom_type,Atom_pos,rota=0,clean=False,Bonds_OH=[],Angles_OH=[]):
    """"""
    lx,Lx = Lims[0]
    ly,Ly = Lims[1]
    lz,Lz = Lims[2]

    Nx_0,Ny_0 = 0,0

    bonds = False
    if len(Bonds_OH) > 0:
        bonds = True

    Pos = []
    Types = []
    Pos_O = Atom_pos[Atom_type==2]
    num_O = np.arange(len(Atom_pos))[Atom_type==2]
    z_prop_add = 0

    #Check if int isntead of list
    flag_x = 0
    flag_y = 0
    if type(Nx_list) is int:
        Nx = Nx_list
        flag_x = 1

    elif len(Nx_list) == 2:
        Nx_0 = Nx_list[0]
        Nx = Nx_list[1]
        flag_x = 1

    else: Nx = Nx_list[-1]

    if type(Ny_list) is int:
        Ny = Ny_list
        flag_y = 1

    elif len(Ny_list) == 2:
        Ny_0 = Ny_list[0]
        Ny = Ny_list[1]
        flag_y = 1

    else: Ny = Ny_list[-1]



    for x in range(Nx_0,Nx):
        for y in range(Ny_0,Ny):
            if flag_x or flag_y or (x >= Nx_list[0] and x< Nx_list[1]) or (x >= Nx_list[2] and x <= Nx_list[3]) or (y >= Ny_list[0] and y < Ny_list[1]) or (y >= Ny_list[2] and y <= Ny_list[3]):
                for z in range(Nz):
                    for at,ty,num in zip(Atom_pos,Atom_type,range(len(Atom_pos))):
                        Pos.append(at + np.array([x*Lx+lx,y*Ly+ly,z*Lz+lz]))
                        Types.append(ty)




    Pos = np.array(Pos).reshape(len(Pos),3)
    min_x,max_x = np.min(Pos[:,0]),np.max(Pos[:,0])
    min_y,max_y = np.min(Pos[:,1]),np.max(Pos[:,1])
    dx, dy  = (max_x), (max_y)
    max_d = max(dx,dy)
    maxx =  max_d * (2**(1/2))
    maxy =  max_d * (2**(1/2))


    Lims_tot = np.array([[-maxx,maxx],[-maxy,maxy],[lz,Lz]])
    z_lim = [lz,Lz*Nz]

    num_at = len(Atom_pos)
    Bonds_OH_tot = []
    Angles_OH_tot = []
    if bonds:
        for j in range(Nz):
            for bond in Bonds_OH:
                Bonds_OH_tot.append([bond[0]+num_at*j,bond[1]+num_at*j])
            for angle in Angles_OH:
                Angles_OH_tot.append([angle[0]+num_at*j,angle[1]+num_at*j,angle[2]+num_at*j])
        return Pos,np.array(Types),Lims_tot, z_lim, Bonds_OH_tot, Angles_OH_tot

    else:
        return Pos,np.array(Types),Lims_tot, z_lim


def transfo(Pos,slide_z=0,D=0,rota=0,enlarge=10,enlarge_z=0.700758,do_periodic=True,circling=True):
    mean = np.mean(Pos,axis=0)
    mean[2] = np.min(Pos[:,2])
    # mean[2] = 0
    # print(mean)
    Pos = Pos - mean

    Lz = np.max(Pos[:,2]) / 2/np.pi
    # print(Lz)
    x,y,z = Pos.transpose()
    z = z/Lz
    # Lz = z_lim[1] - z_lim[0]


    # theta = 0.5
    # y = y*np.cos(theta) - z*np.sin(theta)
    # z = y*np.sin(theta) + z*np.cos(theta)


    # z_coord = (Lz**2-rota**2*np.pi**2*D**2)**(1/2) *(z-z_lim[0])/Lz
    Lx = np.max(x)
    lx = np.min(x)
    Ly = np.max(y)
    ly = np.min(y)
    LX = Lx-lx
    LY = Ly-ly



    if circling:
        x = (x-lx - LX/2) / LX * 2
        y = (y-ly - LY/2) / LY * 2
        #FG Squircular Mapping
        # x_coord = x * (x**2 + y**2 - x**2*y**2)**(1/2) / (x**2+y**2)**(1/2)
        # y_coord = y * (x**2 + y**2 - x**2*y**2)**(1/2) / (x**2+y**2)**(1/2)
        #Simple Strecthing
        # x_coord = np.sign(x)*(x**2 / (x**2+y**2)**(1/2)) * (x**2>=y**2) + np.sign(y)*(x*y)/(x**2+y**2)**(1/2) * (x**2 < y**2)
        # y_coord = np.sign(x)*(x*y / (x**2+y**2)**(1/2)) * (x**2>=y**2) + np.sign(y)*(y*y)/(x**2+y**2)**(1/2) * (x**2 < y**2)
        #Elliptical Mapping
        x_coord = x * (1-1/2*y**2)**(1/2)
        y_coord = y * (1-1/2*x**2)**(1/2)


        # #Schwarz Christoffel Mapping
        # import scipy.special as sps
        # import mpmath
        # # print(sps.ellipj(x+1j*y,(1/2)**(1/2)))
        # x_coord = []
        # y_coord = []
        # for xa,ya in zip(x,y):
        #     c = (1-1j)/(2)**(1/2) * mpmath.ellipfun("cn",1.854 * (1+1j)/2 * (xa+1j*ya)-1.854,(1/2))
        #     x_coord.append(float(c.real))
        #     y_coord.append(float(c.imag))
        #
        #
        # x_coord = np.array(x_coord)
        # y_coord = np.array(y_coord)

        x = x_coord * Lx
        y = y_coord * Ly

    # x_coord = x
    # y_coord = y
    # x_coord = (x) * np.cos(2*np.pi*rota*(z/Lz)) - (y+D) * np.sin(2*np.pi*rota*(z/Lz))
    # y_coord = (x) * np.sin(2*np.pi*rota*(z/Lz)) + (y+D) * np.cos(2*np.pi*rota*(z/Lz))-D

    R = D * rota
    Norm = (Lz**2 + R**2)**(1/2)
    z_coord = Lz * z + R * x / Norm
    # z_coord = Lz * z
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
    Pos_transfo = np.array([x_coord + mean[0],y_coord + mean[1],z_coord]).transpose()
    Lims = np.array([[np.min(Pos_transfo[:,0]-enlarge),np.max(Pos_transfo[:,0])+enlarge],[np.min(Pos_transfo[:,1]-enlarge),np.max(Pos_transfo[:,1])+enlarge],[0,np.max(Pos_transfo[:,2])+enlarge_z]])

    return Pos_transfo,Lims, slide_z



def write_data(file_name,Pos,Types,Lims,D=0,Bonds_OH=[],Angles_OH=[]):
    H_present = False
    if 3 in Types:
        H_present = True

    bonds = False
    if len(Bonds_OH) > 0:
        bonds = True

    with open(file_name,"w") as file:
        file.write("\n")
        file.write("{} atoms".format(len(Types)))
        file.write("\n")

        if bonds and H_present:
            file.write("{} bonds".format(len(Bonds_OH)))
            file.write("\n")
            file.write("{} angles".format(len(Angles_OH)))
            file.write("\n")
        file.write("\n")

        file.write("{} atom types".format(np.max(Types)))
        file.write("\n")

        if bonds and H_present:
            file.write("1 bond types")
            file.write("\n")
            file.write("1 angle types")
            file.write("\n")

        file.write("\n")
        file.write("{:3.6f} {:3.6f} xlo xhi".format(Lims[0][0],Lims[0][1]))
        file.write("\n")
        file.write("{:3.6f} {:3.6f} ylo yhi".format(Lims[1][0],Lims[1][1]))
        file.write("\n")
        file.write("{:3.6f} {:3.6f} zlo zhi".format(Lims[2][0],Lims[2][1]))
        file.write("\n")
        file.write("\n")

        file.write("Masses")
        file.write("\n")
        file.write("\n")
        file.write("1 28.0855")
        file.write("\n")
        file.write("2 15.9994")
        file.write("\n")

        if H_present:
            file.write("3 15.9994")
            file.write("\n")
            file.write("4 1.0080")
            file.write("\n")

        file.write("\n")
        file.write("Atoms")
        file.write("\n")
        file.write("\n")

        for num,pos,typ in zip(range(len(Pos)),Pos,Types):
            num = str(num+1)
            file.write(num + " "*(8-len(num)) + "1 {} 0.0 {:3.6f} {:3.6f} {:3.6f}\n".format(typ,pos[0],pos[1],pos[2]))

        if bonds and H_present:
            file.write("\n")
            file.write("Bonds")
            file.write("\n")
            file.write("\n")
            for num,bond in zip(range(len(Bonds_OH)),Bonds_OH):
                file.write("{} 1 {} {}\n".format(num+1,bond[0],bond[1]))

            file.write("\n")
            file.write("Angles")
            file.write("\n")
            file.write("\n")
            for num,bond in zip(range(len(Angles_OH)),Angles_OH):
                file.write("{} 1 {} {} {}\n".format(num+1,bond[0],bond[1],bond[2]))


def clean_structure(Pos,Types,Lims,N,periodic=True):
    Center = (Lims[0][1]*N[0][0] - Lims[0][0])/2

    Bonds, Si_count_O, O_count_Si = compute_bonds(Pos,Types,Lims=Lims*N,periodic=periodic)[:3]

    Pos_Si = Pos[Types==1]
    Pos_O = Pos[Types==2]



    Lack_Si = Pos_Si[Si_count_O == 2]



    Atoms_add_pos = []
    Atoms_add_types = []
    Bonds_OH = []
    num_at = len(Pos)

    for Si in Lack_Si:
        Atom_O_add = np.sign(Si[0]-Center) * np.array([1.6,0,0]) + Si
        Atom_H_add = np.sign(Si[0]-Center) * np.array([2.6,0,0]) + Si

        Atoms_add_pos.append(Atom_O_add)
        Atoms_add_pos.append(Atom_H_add)
        Atoms_add_types.append(3)
        Atoms_add_types.append(4)
        Bonds_OH.append([num_at,num_at+1])


        Pos_O_trunc = Pos_O[(Pos_O[:,2] > (Si[2]-2)) * (Pos_O[:,2] < (Si[2]+2))]
        Si_symm = np.array([2*Center-Si[0],Si[1],Si[2]])

        #To rewrite for non orthogonal
        Dist_O = np.sum((Pos_O_trunc - Si_symm.reshape((1,3)))**2,axis=1)
        Pos_O_add_H = Pos_O_trunc[np.argmin(Dist_O)]

        Atom_H_add = np.sign(Pos_O_add_H-Center) * np.array([1.0,0,0]) + Pos_O_add_H
        Atoms_add_pos.append(Atom_H_add)
        Atoms_add_types.append(4)


        Index_O_trunc = (Types==2) * (Pos[:,2] > (Si[2]-2)) * (Pos[:,2] < (Si[2]+2))
        Index_O_bond = np.argmax((np.cumsum(Index_O_trunc)-1)==np.argmin(Dist_O))
        # print(Pos_O_add_H,Pos[Index_O_bond])
        # print(np.argmin(np.linalg.norm(Pos[Index_O_trunc] - Pos_O_add_H,axis=1)),Index_O_bond)
        Types[Index_O_bond] = 3

        Bonds_OH.append([Index_O_bond,num_at+2])

        num_at += 3

    New_Pos, New_Types = np.append(Pos,Atoms_add_pos,axis=0), np.append(Types,Atoms_add_types,axis=0)
    Middle = (New_Pos[:,2] >= (Lims[2,1]+Lims[2,0])) * (New_Pos[:,2] < (Lims[2,1]*2+Lims[2,0]))
    New_Pos = New_Pos[Middle]
    New_Pos = New_Pos - np.array([0,0,np.min(New_Pos[:,2])])
    New_Types = New_Types[Middle]

    Bonds_OH_corrected = []
    New_index = (np.cumsum(Middle)-1) * Middle

    # print(Bonds_OH)
    for bond in Bonds_OH:
        if New_index[bond[0]] != 0 and New_index[bond[1]] != 0:
            Bonds_OH_corrected.append([New_index[bond[0]]+1,New_index[bond[1]]+1])


    #Bonds : O then H
    Angles_OH = []
    Pos_Si = New_Pos[New_Types==1]
    for bond in Bonds_OH_corrected:
        O = New_Pos[bond[0]-1]
        Pos_Si_trunc =  Pos_Si[(Pos_Si[:,2] > (O[2]-2)) * (Pos_Si[:,2] < (O[2]+2))]
        Dist_Si = np.sum((Pos_Si_trunc - O.reshape((1,3)))**2,axis=1)

        Index_Si_trunc = (New_Types==1) * (New_Pos[:,2] > (O[2]-2)) * (New_Pos[:,2] < (O[2]+2))
        Index_Si_bond = np.argmax((np.cumsum(Index_Si_trunc)-1) == np.argmin(Dist_Si))
        Angles_OH.append([Index_Si_bond+1,bond[0],bond[1]])



    return New_Pos, New_Types, Bonds_OH_corrected, Angles_OH





def create_syst(rota,D,pitch,width,thickness,int_thick,do_correct=True,do_periodic=True,circling=True):
    Lims, Atom_types, Atom_pos = read_data("quartz_clean_test.data")

    lx = np.max(Atom_pos[:,0]) - np.min(Atom_pos[:,0])
    ly = np.max(Atom_pos[:,1]) - np.min(Atom_pos[:,1])
    lz = np.max(Atom_pos[:,2]) - np.min(Atom_pos[:,2])

    Nx = int(width // lx +1)
    Ny = int(thickness // ly +1)
    Nz = int(pitch // lz +1)

    Nx_int = int(int_thick // lx +1)
    Ny_int = int(int_thick // ly +1)

    Nx_list = [0,Nx_int,Nx-Nx_int,Nx]
    Ny_list = [0,Ny_int,Ny-Ny_int,Ny]

    N_list = np.array([[Nx],[Ny],[Nz]])

    if int_thick == 0:
        Nx_list = Nx
        Ny_list = Ny

    print(Nx_list,Ny_list,Nz)

    ##Ext

    if do_correct:
        Pos, Types, Lims_tot, z_lim = duplicate(Nx_list,Ny_list,3,Lims,Atom_types,Atom_pos,rota=rota)
        Pos, Types, Bonds_OH, Angles_OH = clean_structure(Pos,Types,Lims,N_list,periodic=True)
        Pos, Types, Lims_tot, z_lim, Bonds_OH, Angles_OH = duplicate(1,1,Nz,Lims,Types,Pos,rota=rota,Bonds_OH=Bonds_OH,Angles_OH=Angles_OH)
    else:
        Pos, Types, Lims_tot, z_lim = duplicate(Nx_list,Ny_list,Nz,Lims,Atom_types,Atom_pos,rota=rota)
        Bonds_OH, Angles_OH = [], []

    Pos_transfo,Lims_tot,slide_z = transfo(Pos,D=D,rota=rota,do_periodic=do_periodic,circling=circling)
    write_data("quartz_dupl.data",Pos_transfo,Types,Lims_tot,D=D,Bonds_OH=Bonds_OH,Angles_OH=Angles_OH)

    if not type(Nx_list) is int:
        ##Int
        Nx_int = Nx_list[1:3]
        Ny_int = Ny_list[1:3]
        Pos_int, Types_int, Lims_tot_int, z_lim = duplicate(Nx_int,Ny_int,Nz,Lims,Atom_types,Atom_pos,rota=rota)
        Pos_transfo_int ,Lims_tot_int,slide_z = transfo(Pos_int,D=D,rota=rota,slide_z=slide_z,do_periodic=do_periodic,circling=circling)
        write_data("quartz_int.data",Pos_transfo_int,Types_int,Lims_tot_int,Bonds_OH=Bonds_OH,Angles_OH=Angles_OH)
    else:
        Pos_transfo_int = None
        Types_int = None
        Lims_tot_int = None

    return Pos_transfo, Types, Lims_tot, Angles_OH, Pos_transfo_int, Types_int, Lims_tot_int





if __name__ == "__main__":
    rota = 0.0
    # rota = 0.0
    # Nx,Ny,Nz = 33,24,99
    # Nx,Ny,Nz = 10,16,80
    # D = 167
    # pitch = 600
    # width = 200
    # thickness = 110
    # int_thick = 35
    # D = 50
    # D = 40
    # D = 0
    # pitch = 300
    # width = 80
    # thickness = 40
    # int_thick = 10
    # D = 80
    # pitch = 300
    # width = 60
    # thickness = 30
    # int_thick = 10
    # #
    D = 0
    pitch = 15
    width = 10
    thickness = 10
    int_thick = 0

    # Nx,Ny,Nz = 4,4,4
    Pos_transfo,Types,Lims_tot,Angles_OH, Pos_transfo_int, Types_int, Lims_tot_int = create_syst(rota,D,pitch,width,thickness,int_thick,do_correct=False,circling=False)
    # Pos_transfo,Types,Lims_tot,Angles_OH, Pos_transfo_int, Types_int, Lims_tot_int = create_syst(rota,D,pitch,width,thickness,int_thick,circling=True)
    print("Number of Si : ",np.sum(Types==1))

    # Pos_O = Pos_transfo[Types==2]
    # Dist = sd.cdist(Pos_O,Pos_O)
    # Dist = Dist + 100*(Dist == 0)
    # print(np.min(Dist,axis=0))
    # write_data("quartz_dupl.data",Pos_transfo,Types,Lims_tot,D=D,Bonds_OH=[],Angles_OH=[])

    if 0:
        import matplotlib.pyplot as plt
        analyze_mult([0],[Pos_transfo],[Types],periodic=True,Lims=Lims_tot)
        # import pyvista as pv
        # analyze_plot_syst(Pos_transfo,Types,periodic=True,Lims=Lims_tot)

    if 1:
        import pyvista as pv
        # Types = Types[ (Pos_transfo[:,2]>5) * (Pos_transfo[:,2]<10)]
        # Pos_transfo = Pos_transfo[(Pos_transfo[:,2]>5) * (Pos_transfo[:,2]<10)]
        # if Types_int:
        #     Types_int = Types_int[(Pos_transfo_int[:,2]<5) * (Pos_transfo_int[:,2]<10)]
        #     Pos_transfo_int = Pos_transfo_int[(Pos_transfo_int[:,2]<5) * (Pos_transfo_int[:,2]<10)]

        Bonds = compute_bonds(Pos_transfo,Types,periodic=False,Lims=Lims_tot)[0]

        plotter = pv.Plotter()
        plotter.add_axes()

        # purple = np.array([96,25,255])/255
        # dark_purple = np.array([56,20,180])/255

        Si_c = [240,200,160]
        O_c = [255,13,13]
        H_c = [255,255,255]

        sp = pv.Sphere(radius=0.4)

        # data = pv.PolyData(Pos_Si.reshape((np.shape(Pos_Si)[0],3)) + np.array([[30,0,0]]))
        # pc = data.glyph(scale=False,geom=sp,orient=False)
        # plotter.add_mesh(pc,opacity=0.7,pbr=True,roughness=.5,metallic=.2,color="red")
        #
        # data = pv.PolyData(Pos_O.reshape((np.shape(Pos_O)[1],3)) + np.array([[30,0,0]]))
        # pc = data.glyph(scale=False,geom=sp,orient=False)
        # plotter.add_mesh(pc,opacity=0.7,pbr=True,roughness=.5,metallic=.2,color="blue")

        data = pv.PolyData(Pos_transfo[Types==1])
        pc = data.glyph(scale=False,geom=sp,orient=False)
        # plotter.add_mesh(pc,opacity=0.05,pbr=True,roughness=.5,metallic=.2,color="red")
        plotter.add_mesh(pc,opacity=0.9,pbr=True,roughness=.5,metallic=.2,color=Si_c)

        data = pv.PolyData(Pos_transfo[Types==2])
        pc = data.glyph(scale=False,geom=sp,orient=False)
        plotter.add_mesh(pc,opacity=0.9,pbr=True,roughness=.5,metallic=.2,color=O_c)

        if np.sum(Types==3):
            data = pv.PolyData(Pos_transfo[Types==3])
            pc = data.glyph(scale=False,geom=sp,orient=False)
            plotter.add_mesh(pc,opacity=0.9,pbr=True,roughness=.5,metallic=.2,color=O_c)

            data = pv.PolyData(Pos_transfo[Types==4])
            pc = data.glyph(scale=False,geom=sp,orient=False)
            plotter.add_mesh(pc,opacity=0.9,pbr=True,roughness=.5,metallic=.2,color=H_c)





        if type(Types_int) is type(np.array([])):
            data = pv.PolyData(Pos_transfo_int[Types_int==1])
            pc = data.glyph(scale=False,geom=sp,orient=False)
            plotter.add_mesh(pc,opacity=1.0,pbr=True,roughness=.5,metallic=.2,color="black")

            data = pv.PolyData(Pos_transfo_int[Types_int==2])
            pc = data.glyph(scale=False,geom=sp,orient=False)
            plotter.add_mesh(pc,opacity=1.0,pbr=True,roughness=.5,metallic=.2,color="green")

        # data = pv.Plane(center=(0,0,0),i_size=100,j_size=100)
        # plotter.add_mesh(data,opacity=0.2,color="red")
        # Lz = 226.4537896205332
        # data = pv.Plane(center=(0,0,Lz),i_size=100,j_size=100)
        # plotter.add_mesh(data,opacity=0.2,color="red")


        N_Si,N_O = np.shape(Bonds)
        Indices = (np.arange(0,N_Si).reshape(N_Si,1,1)*np.array([1,0]) + np.arange(0,N_O).reshape((1,N_O,1))*np.array([0,1])).reshape((N_Si*N_O,2))
        Pos_Si = Pos_transfo[Types==1]
        Pos_O = Pos_transfo[(Types==2)+(Types==3)]

        Indices = Indices[Bonds.ravel()!=0]

        tubes = [pv.Tube(Pos_Si[i_si],Pos_O[i_o],n_sides=5,radius=0.2) for i_si,i_o in Indices]
        mesh = tubes[0].merge(tubes[1:])

        plotter.add_mesh(mesh,opacity=0.3,pbr=True,roughness=.5,metallic=.2,color="white")

        # tubes = []
        # for k in Angles_OH:
        #     tubes.append(pv.Tube(Pos_transfo[k[0]-1],Pos_transfo[k[1]-1],n_sides=5,radius=0.2))
        #     tubes.append(pv.Tube(Pos_transfo[k[1]-1],Pos_transfo[k[2]-1],n_sides=5,radius=0.2))
        # mesh = tubes[0].merge(tubes[1:])
        # #
        # plotter.add_mesh(mesh,opacity=1.0,pbr=True,roughness=.5,metallic=.2,color="blue")




        # data = pv.PolyData(Pos_transfo[Types==3])
        # pc = data.glyph(scale=False,geom=sp,orient=False)
        # plotter.add_mesh(pc,opacity=1.0,pbr=True,roughness=.5,metallic=.2,color="green")
        plotter.show()