import numpy as np
# import matplotlib.pyplot as plt


def read_dump(dump_file,unscale=False):
    """Suppose ATOM dump"""
    flag_step = 0
    flag_num_at = 0
    flag_box_bound = 0
    flag_atoms = 0
    list_TSTEP = []
    list_NUM_AT = []
    list_BOX = []
    list_ATOMS = []
    for line in open(dump_file,"r"):
        if flag_step:
            list_TSTEP.append(int(line))
            flag_step = 0

        elif flag_num_at:
            list_NUM_AT.append(int(line))
            flag_num_at = 0

        elif flag_box_bound:
            lsplit = line.split()
            BOX.append([float(lsplit[0]),float(lsplit[1])])
            flag_box_bound -= 1
            if not flag_box_bound:
                list_BOX.append(BOX)

        elif flag_atoms:
            if "ITEM: TIMESTEP" in line:
                flag_step = 1
                flag_atoms = 0
                list_ATOMS.append(list_at_t)
            else:
                lsplit = line.split()
                if len(lsplit) != 5:
                    raise TypeError("This function suppose that the dump has used the atom type")
                if unscale:
                    Lx = BOX[0][1] - BOX[0][0]
                    Ly = BOX[1][1] - BOX[1][0]
                    Lz = BOX[2][1] - BOX[2][0]
                    list_at_t.append([int(lsplit[0]),int(lsplit[1]),float(lsplit[2])*Lx+BOX[0][0],float(lsplit[3])*Ly+BOX[1][0],float(lsplit[4])*Lz+BOX[2][0]])
                else: list_at_t.append([int(lsplit[0]),int(lsplit[1]),float(lsplit[2]),float(lsplit[3]),float(lsplit[4])])

        elif "ITEM: TIMESTEP" in line:
            flag_step = 1
        elif "ITEM: NUMBER OF ATOMS" in line:
            flag_num_at = 1
        elif "ITEM: BOX BOUNDS" in line:
            flag_box_bound = 3
            BOX = []
        elif "ITEM: ATOMS" in line:
            flag_atoms = 1
            list_at_t = []
    list_ATOMS.append(list_at_t)
    try:
        list_ATOMS = np.array(list_ATOMS)
    except:
        print("Atoms lost or removed during the dynamic. Will only extract the last timestep")
        list_TSTEP = [list_TSTEP[-1]]
        list_NUM_AT = [list_NUM_AT[-1]]
        list_BOX = [list_BOX[-1]]
        list_ATOMS = np.array([list_ATOMS[-1]])

    C_min = np.mean(list_ATOMS[:,:,2:],axis=1) + np.array([Lx/2,Ly/2,Lz/2])
    C_min[:,2] = 0

    # print(np.shape(C_min))
    C_min = C_min.reshape((len(C_min),1,3))
    list_ATOMS[:,:,2:] = list_ATOMS[:,:,2:] - C_min
    list_ATOMS[:,:,2] = list_ATOMS[:,:,2] % Lx
    list_ATOMS[:,:,3] = list_ATOMS[:,:,3] % Ly
    list_ATOMS[:,:,4] = (list_ATOMS[:,:,4]-BOX[2][0]) % Lz + BOX[2][0]
    # Pos_Z = list_ATOMS[:,:,4]
    # list_ATOMS[:,:,4] = (Pos_Z-Lz) * (Pos_Z > Lz) + (Pos_Z-Lz) * (Pos_Z > Lz)

    return list_TSTEP, list_NUM_AT, list_BOX, np.array(list_ATOMS)


def write_dump(file_name,list_TSTEP,list_NUM_AT,list_BOX,list_ATOMS):
    with open(file_name,"w") as file:
        for tstep, num_at, box, atoms in zip(list_TSTEP,list_NUM_AT,list_BOX,list_ATOMS):

            Lx = box[0][1] - box[0][0]
            Ly = box[1][1] - box[1][0]
            Lz = box[2][1] - box[2][0]

            file.write("ITEM: TIMESTEP")
            file.write("\n")
            file.write(str(tstep))
            file.write("\n")
            file.write("ITEM: NUMBER OF ATOMS")
            file.write("\n")
            file.write(str(num_at))
            file.write("\n")
            file.write("ITEM: BOX BOUNDS pp pp pp")
            file.write("\n")
            for box_c in box:
                file.write("{:3.6f} {:3.6f}".format(box_c[0],box_c[1]))
                file.write("\n")
            file.write("ITEM: ATOMS id type xs ys zs")
            file.write("\n")
            for at in atoms:
                file.write("{} {} {:3.6f} {:3.6f} {:3.6f}\n".format(int(at[0]),int(at[1]),at[2]/Lx,at[3]/Ly,at[4]/Lz))





def read_data(file,do_scale=True):
    """yes"""
    BOX = []
    list_at_t = []
    for line in open(file,"r"):
        lsplit = line.split()
        if "xlo" in line or "ylo" in line or "zlo" in line:
            BOX.append([float(lsplit[0]),float(lsplit[1])])

        if len(lsplit) == 7:
            Lx = BOX[0][1] - BOX[0][0]
            Ly = BOX[1][1] - BOX[1][0]
            Lz = BOX[2][1] - BOX[2][0]
            if do_scale:
                list_at_t.append([int(lsplit[0]),int(lsplit[2]),float(lsplit[4])/Lx,float(lsplit[5])/Ly,float(lsplit[6])/Lz])
            else: list_at_t.append([int(lsplit[0]),int(lsplit[2]),float(lsplit[4]),float(lsplit[5]),float(lsplit[6])])
    list_ATOMS = [list_at_t]
    list_BOX = [BOX]

    return list_BOX, np.array(list_ATOMS)






def create_distorted_data(file,N=(10,10,10),N_r=20,D=10,periodic=False):
    if ".data" in file: list_BOX, list_ATOMS = read_data(file)
    else: list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS = read_dump(file)
    tx,Tx = list_BOX[0][0]
    ty,Ty = list_BOX[0][1]
    tz,Tz = list_BOX[0][2]

    TLx,TLy,TLz = Tx-tx,Ty-ty,Tz-tz

    last_ATOMS = list_ATOMS[-1]
    list_Type = last_ATOMS[:,1] != 3


    list_X = np.sort(last_ATOMS[:,2][list_Type])
    list_Y = np.sort(last_ATOMS[:,3][list_Type])
    list_Z = np.sort(last_ATOMS[:,4][list_Type])


    ax, x_min = np.polyfit(np.arange(len(list_X)),list_X,1)
    ay, y_min = np.polyfit(np.arange(len(list_Y)),list_Y,1)
    az, z_min = np.polyfit(np.arange(len(list_Z)),list_Z,1)

    x_max = ax*len(list_X) + x_min
    y_max = ay*len(list_Y) + y_min
    z_max = az*len(list_Z) + z_min


    Lx = (x_max-x_min)*TLx
    Ly = (y_max-y_min)*TLy
    Lz = (z_max-z_min)*TLz


    x_min = x_min * TLx + tx
    y_min = y_min * TLy + ty
    z_min = z_min * TLz + tz

    nx, ny, nz = N

    rota = 0.0
    for k in range(N_r):
        rota +=1/N_r
        with open("deformation_{}.data".format(k),"w") as file:
            file.write("\n")
            file.write("{} atoms".format(nx*ny*nz))
            file.write("\n")
            file.write("1 atom types")

            file.write("\n")
            file.write("\n")

            file.write("Masses")
            file.write("\n")
            file.write("\n")
            file.write("3 100000.0")

            file.write("\n")
            file.write("\n")
            file.write("Atoms")
            file.write("\n")
            file.write("\n")

            for z in range(1,nz+1):
                z_coord = (Lz**2-rota**2*np.pi**2*D**2)**(1/2) *(z-1)/(nz-1) + z_min
                if periodic:
                    for x in range(nx):
                        for y in range(ny):
                            num = str(z+y*nz+x*ny*nz)

                            x_cube = (Lx)*(x)/(nx-1)-Lx/2 + D
                            y_cube = (Ly)*(y)/(ny-1)-Ly/2

                            if z<(nz//2):
                                x_coord = x_cube * np.cos(2*np.pi*rota*(z_coord/Lz/2)) - y_cube * np.sin(2*np.pi*rota*(z_coord/Lz/2)) + Lx/2 - D
                                y_coord = x_cube * np.sin(2*np.pi*rota*(z_coord/Lz/2)) + y_cube * np.cos(2*np.pi*rota*(z_coord/Lz/2)) + Ly/2

                            else:
                                x_coord = x_cube * np.cos(2*np.pi*(1-rota*(z_coord/Lz/2))) - y_cube * np.sin(2*np.pi*(1-rota*(z_coord/Lz/2))) + Lx/2 - D
                                y_coord = x_cube * np.sin(2*np.pi*(1-rota*(z_coord/Lz/2))) + y_cube * np.cos(2*np.pi*(1-rota*(z_coord/Lz/2))) + Ly/2

                            file.write(num + " "*(8-len(num)) + "3 0.0 {:3.6f} {:3.6f} {:3.6f}\n".format(x_coord,y_coord,z_coord))



                else:
                    for x in range(nx):
                        for y in range(ny):
                            num = str(z+y*nz+x*ny*nz)

                            x_cube = (Lx)*(x)/(nx-1)-Lx/2 + D
                            y_cube = (Ly)*(y)/(ny-1)-Ly/2
                            # z_coord = (Lz)*(z-1)/(nz-1)

                            x_coord = x_cube * np.cos(2*np.pi*rota*(z_coord/Lz)) - y_cube * np.sin(2*np.pi*rota*(z_coord/Lz)) + Lx/2 - D
                            y_coord = x_cube * np.sin(2*np.pi*rota*(z_coord/Lz)) + y_cube * np.cos(2*np.pi*rota*(z_coord/Lz)) + Ly/2
                            # x_coord = x_cube * np.cos(2*np.pi*rota*(z/nz-1)) - y_cube * np.sin(2*np.pi*rota*((z-1)/(nz-1)))
                            # y_coord = x_cube * np.sin(2*np.pi*rota*(z/nz-1)) + y_cube * np.cos(2*np.pi*rota*((z-1)/(nz-1)))

                            file.write(num + " "*(8-len(num)) + "3 0.0 {:3.6f} {:3.6f} {:3.6f}\n".format(x_coord,y_coord,z_coord))

if __name__ == "__main__":
    # list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS = read_dump("dumps_compiled.lammpstrj")
    # write_dump("dumps_trimmed.lammpstrj",list_TSTEP[::10], list_NUM_AT[::10], list_BOX[::10], list_ATOMS[::10])
    list_BOX,list_ATOMS = read_data("quartz_dupl.data",do_scale=False)
    write_dump("quartz_dupl.lammpstrj",[0], [len(list_ATOMS[0])], list_BOX, list_ATOMS)



