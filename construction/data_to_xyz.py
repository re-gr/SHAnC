import numpy as np
from create_distorted import read_dump

file = "dummps_long_last.lammpstrj"

list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS = read_dump(file)



tx,Tx = list_BOX[0][0]
ty,Ty = list_BOX[0][1]
tz,Tz = list_BOX[0][2]

Lx = Tx-tx
Ly = Ty-ty
Lz = Tz-tz

last_ATOMS = list_ATOMS[-1]
list_Type = (last_ATOMS[:,1] != 5) * (last_ATOMS[:,1] != 6)

list_ATOMS_sel = last_ATOMS[list_Type]

with open(file.split(".")[0] + ".xyz","w") as f:
    f.write(str(len(list_ATOMS_sel)))
    f.write("\n")
    f.write("\n")
    for num,type_at,x,y,z in list_ATOMS_sel:
        if type_at == 1:
            type_write = "Si "
        elif type_at == 4:
            type_write = "H "
        else: type_write = "O "
        xs = x*Lx + tx
        ys = y*Ly + ty
        zs = z*Lz + tz

        f.write(type_write + " {:3.6f} {:3.6f} {:3.6f}\n".format(xs,ys,zs))




