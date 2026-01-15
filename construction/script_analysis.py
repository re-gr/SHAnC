import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from anim_disto import *
from create_distorted import *
import pyvista as pv
import scipy.spatial.distance as sd
import scipy.sparse.csgraph as sps



def compute_bonds(Pos,Types,threshold_Si=2,threshold_O=2,threshold_H=1.3,do_count_type_3=True):
    """
    compute_bonds(Pos,Types,threshold_Si=2,threshold_O=2,threshold_H=1.3,do_count_type_3=True)

    Compute the Bonds of a silica system.
    |!| This version must only be used for small systems. It takes too much memory otherwise.
    The types are the following : 1 : Si, 2: O, 3: Oh, 4: H

    Parameters
    ----------
    Pos : array
        The position of the atoms
    Types : array
        The type of the Atoms
    threshold_Si : float, optional
        The threshold used to consider if Si and O are bonding. 2 by default
    threshold_O : float, optional
        The threshold used to consider if O and Si are bonding. 2 by default
    threshold_H : float, optional
        The threshold used to consider if O and H are bonding. 1.3 by default
    do_count_type_3 : bool, optional
        If one wants to consider the Oh in the calculations.

    Returns
    -------
        Bonds, Si_count_O, O_count_Si, O_count_H, H_count_O
    """

    Pos_Si = Pos[Types==1]
    if do_count_type_3:
        Pos_O = Pos[((Types==2) + (Types==3)).astype("bool")]
    else:
        Pos_O = Pos[ ((Types==2)).astype("bool")]
    Pos_H = Pos[Types==4]
    if Pos_H.any(): H_present = True
    else: H_present = False

    num_at = len(Types)
    num_Si = len(Pos_Si)
    num_O = len(Pos_O)

    # There are multiple ways to compute the distance, but this one is the fastest I found
    Dist = sd.cdist(Pos_Si,Pos_O)

    # Pos_Si = Pos_Si.reshape(len(Pos_Si),1,3)
    # Pos_O = Pos_O.reshape(1,len(Pos_O),3)
    # Pos_H = Pos_H.reshape((len(Pos_H),1,3))

    # Dist = (Pos_Si - Pos_O) **2
    # Dist = np.einsum("ilj->il",Dist)**(1/2)


    Si_Nearest_O = np.min(Dist,axis=1).reshape((len(Dist),1))
    O_Nearest_Si = np.min(Dist,axis=0)

    Bonds_Si = (Dist<(threshold_Si))
    Bonds_O = (Dist<(threshold_O))
    # Bonds_Si = Bonds_Si - 1*(Si_Nearest_O > threshold_O)
    # Bonds_O = Bonds_O - 1*(O_Nearest_Si > threshold_Si)
    Bonds =  Bonds_Si + Bonds_O


    Si_count_O = np.sum(Bonds,axis=1)
    O_count_Si = np.sum(Bonds,axis=0)

    if H_present:
        Dist_OH = sd.cdist(Pos_H,Pos_O[:num_O])

        Bonds_OH = Dist_OH<threshold_H
        O_count_H = np.sum(Bonds_OH,axis=0)
        H_count_O = np.sum(Bonds_OH,axis=1)
    else : O_count_H, H_count_O = np.array([]),np.array([])

    return Bonds, Si_count_O, O_count_Si, O_count_H, H_count_O


def compute_bonds_graph(Pos,Types,cube=30,threshold_Si=2,threshold_O=2,threshold_H=1.3,periodic=True,Lims=[]):
    """
    compute_bonds_graph(Pos,Types,cube=30,threshold_Si=2,threshold_O=2,threshold_H=1.3,periodic=True,Lims=[])

    Compute the bonds that can be used to create a graph.
    The output is a neighbor matrix array taht can be converted to a graph using networkx nx.from_numpy_array

    Parameters
    ----------

    Pos : array
        The position of the atoms of the system
    Types: array
        The types of the atoms of the system
    cube: float, optional
        The edge of the cubes used to divide the system, 30 by default. Larger cubes are faster but are more memory intensive
    threshold_Si : float, optional
        The threshold used to consider if Si and O are bonding. 2 by default
    threshold_O : float, optional
        The threshold used to consider if O and Si are bonding. 2 by default
    threshold_H : float, optional
        The threshold used to consider if O and H are bonding. 1.3 by default
    periodic : bool, optional
        Compute as if the system was periodic, True by default
    Lims : list, optional
        The limits of the system. It is necessary for periodic computations

    Returns
    -------
        Neighbor matrix of the system

    """

    Lx,Ly,Lz = np.max(Pos,axis=0)
    lx,ly,lz = np.min(Pos,axis=0)
    if periodic:
        if len(Lims) == 0:
            print("No limits were provided, the system will be taken as NON PERIODIC. Use the Lim keyword to add limits")
            periodic = False
        else:
            lz,Lz = Lims[2]


    Nx = int((Lx - lx) // cube + 1)
    Ny = int((Ly - ly) // cube + 1)
    Nz = int((Lz - lz) // cube + 1)

    Pos_added = np.copy(Pos)
    Types_added = np.copy(Types)

    Num_Si_or = np.sum(Types==1)


    if periodic:
        #Adds atoms to the system periodically to account for the periodicity

        Dz = Lz-lz

        Pos_add_z = Pos[:,2] > (Lz - rdf_max)
        Pos_remove_z = Pos[:,2] < (lz + rdf_max)

        Pos_add_Lz = Pos[Pos_add_z] - np.array([[0,0,Dz]])
        Pos_remove_Lz = Pos[Pos_remove_z] + np.array([[0,0,Dz]])

        Pos_add = np.append(Pos_add_Lz,Pos_remove_Lz,axis=0)
        Pos_added = np.append(Pos_added,Pos_add,axis=0)

        Types_add = np.append(Types[Pos_add_z],Types[Pos_remove_z],axis=0)
        Types_added = np.append(Types_added,Types_add,axis=0)

    # print("MIN")
    # print(np.min(Pos_added[:,0]),np.max(Pos_added[:,0]))
    # print(np.min(Pos_added[:,1]),np.max(Pos_added[:,1]))
    # print(np.min(Pos_added[:,2]),np.max(Pos_added[:,2]))

    Num_Si = np.sum(Types_added==1)
    Bonds_tot = np.zeros((Num_Si,Num_Si))
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                # print("LIMS")
                # print(((x*cube + lx - threshold_Si - 0.2)),((x+1)*cube + lx + threshold_Si + 0.2))
                # print(((y*cube + ly - threshold_Si - 0.2)),((y+1)*cube + ly + threshold_Si + 0.2))
                # print(((z*cube + lz - threshold_Si - 0.2)),((z+1)*cube + lz + threshold_Si + 0.2))

                #Slice the system inside this cube
                Pos_trunc_x = (Pos_added[:,0] > (x*cube + lx - threshold_Si - 0.2)) * (Pos_added[:,0] < ((x+1)*cube + lx + threshold_Si + 0.2))
                Pos_trunc_y = (Pos_added[:,1] > (y*cube + ly - threshold_Si - 0.2)) * (Pos_added[:,1] < ((y+1)*cube + ly + threshold_Si + 0.2))
                Pos_trunc_z = (Pos_added[:,2] > (z*cube + lz - threshold_Si - 0.2)) * (Pos_added[:,2] < ((z+1)*cube + lz + threshold_Si + 0.2))
                Pos_trunc_ind = Pos_trunc_x * Pos_trunc_y * Pos_trunc_z
                Pos_trunc = Pos_added[Pos_trunc_ind]
                Types_trunc = Types_added[Pos_trunc_ind]
                if (Types_trunc == 1).any() and (Types_trunc==2).any():

                    Bonds = compute_bonds(Pos_trunc,Types_trunc,threshold_Si=threshold_Si,threshold_O=threshold_O,threshold_H=threshold_H)[0]
                    Bonds = Bonds.astype("float")
                    #Get Bonds Si
                    Bonds = Bonds.dot(Bonds.transpose())
                    #Set distance to 1
                    Bonds = Bonds / (Bonds + (Bonds==0)*1)
                    Pos_trunc_ind_Si = Pos_trunc_ind[Types_added==1]

                    Pos_trunc_ind_Si = np.matmul(Pos_trunc_ind_Si.reshape((len(Pos_trunc_ind_Si),1)),Pos_trunc_ind_Si.reshape((1,len(Pos_trunc_ind_Si))))

                    Bonds_tot[Pos_trunc_ind_Si] = Bonds_tot[Pos_trunc_ind_Si] + Bonds.ravel()
    Bonds_tot_or = Bonds_tot[:Num_Si_or]


    if periodic:
        Pos_add_z_Si = Pos_add_z[Types==1]
        Pos_remove_z_Si = Pos_remove_z[Types==1]
        num_add = np.sum(Pos_add_z_Si)

        Bonds_tot_or[Pos_add_z_Si] = Bonds_tot_or[Pos_add_z_Si] + Bonds_tot[Num_Si_or:Num_Si_or+num_add]
        Bonds_tot_or[Pos_remove_z_Si] = Bonds_tot_or[Pos_remove_z_Si] + Bonds_tot[Num_Si_or+num_add:]
        Bonds_tot_or = Bonds_tot_or[:,:Num_Si_or]

    Bonds_tot_or = Bonds_tot_or + Bonds_tot_or.transpose()
    Bonds_tot_or = Bonds_tot_or / (Bonds_tot_or + (Bonds_tot_or==0)*1)
    Bonds_tot_or = Bonds_tot_or - np.eye(len(Bonds_tot_or)) * Bonds_tot_or
    return Bonds_tot_or

def compute_hist_neighbors(Pos,Types,cube=100,threshold_Si=2,threshold_O=2,threshold_H=1.3,periodic=True,Lims=[],rdf_max=5):
    """
    compute_hist_neighbors(Pos,Types,cube=100,threshold_Si=2,threshold_O=2,threshold_H=1.3,periodic=True,Lims=[],rdf_max=5)

    Computes the Bonds and RDF by slicing the system in multiple subsystems

    Parameters
    ----------

    Pos : array
        The position of the atoms of the system
    Types: array
        The types of the atoms of the system
    cube: float, optional
        The edge of the cubes used to divide the system, 30 by default. Larger cubes are faster but are more memory intensive
    threshold_Si : float, optional
        The threshold used to consider if Si and O are bonding. 2 by default
    threshold_O : float, optional
        The threshold used to consider if O and Si are bonding. 2 by default
    threshold_H : float, optional
        The threshold used to consider if O and H are bonding. 1.3 by default
    periodic : bool, optional
        Compute as if the system was periodic, True by default
    Lims : list, optional
        The limits of the system. It is necessary for periodic computations
    rdf_max : float, optional
        The maximum distance for the RDF. 5 by default

    Returns
    -------
        Dist_list, Si_count_O_tot, O_count_Si_tot

    """

    Lx,Ly,Lz = np.max(Pos,axis=0)
    lx,ly,lz = np.min(Pos,axis=0)
    if periodic:
        if len(Lims) == 0:
            print("No limits were provided, the system will be taken as NON PERIODIC. Use the Lim keyword to add limits")
            periodic = False
        else:
            lz,Lz = Lims[2]

    Nx = int((Lx - lx) // cube + 1)
    Ny = int((Ly - ly) // cube + 1)
    Nz = int((Lz - lz) // cube + 1)

    Pos_added = np.copy(Pos)
    Types_added = np.copy(Types)

    Num_Si_or = np.sum(Types==1)


    if periodic:
        Pos_add_z = Pos[:,2] > (Lz - rdf_max)
        Pos_remove_z = Pos[:,2] < (lz + rdf_max)

        Pos_add_Lz = Pos[Pos_add_z] - np.array([[0,0,Lz]])
        Pos_remove_Lz = Pos[Pos_remove_z] + np.array([[0,0,Lz]])

        Pos_add = np.append(Pos_add_Lz,Pos_remove_Lz,axis=0)
        Pos_added = np.append(Pos_added,Pos_add,axis=0)

        Types_add = np.append(Types[Pos_add_z],Types[Pos_remove_z],axis=0)
        Types_added = np.append(Types_added,Types_add,axis=0)


    Num_at = len(Types)
    Num_Si = np.sum(Types==1)
    Num_O = np.sum(Types==2)
    In_trunc = np.array([1]*Num_at + [0]*(len(Pos_added)-Num_at),dtype="bool")


    Si_count_O_tot = np.zeros((Num_Si))
    O_count_Si_tot = np.zeros((Num_O))
    Dist_list = []
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                #the first slicing is for the system that will the computation will be done to
                Pos_trunc_x_u = (Pos_added[:,0] >= (x*cube + lx)) * (Pos_added[:,0] < ((x+1)*cube + lx))
                Pos_trunc_y_u = (Pos_added[:,1] >= (y*cube + ly)) * (Pos_added[:,1] < ((y+1)*cube + ly))
                Pos_trunc_z_u = (Pos_added[:,2] >= (z*cube + lz)) * (Pos_added[:,2] < ((z+1)*cube + lz))
                Ind_trunc_uniq = Pos_trunc_x_u * Pos_trunc_y_u * Pos_trunc_z_u
                Pos_trunc_uniq = Pos_added[Ind_trunc_uniq]
                Types_trunc_uniq = Types_added[Ind_trunc_uniq]

                #The second slicing is for the RDF, as it needs a larger distance
                Pos_trunc_x = (Pos_added[:,0] >= (x*cube + lx - rdf_max)) * (Pos_added[:,0] < ((x+1)*cube + lx + rdf_max))
                Pos_trunc_y = (Pos_added[:,1] >= (y*cube + ly - rdf_max)) * (Pos_added[:,1] < ((y+1)*cube + ly + rdf_max))
                Pos_trunc_z = (Pos_added[:,2] >= (z*cube + lz - rdf_max)) * (Pos_added[:,2] < ((z+1)*cube + lz + rdf_max))
                Pos_trunc_ind = Pos_trunc_x * Pos_trunc_y * Pos_trunc_z
                Pos_trunc = Pos_added[Pos_trunc_ind]
                Types_trunc = Types_added[Pos_trunc_ind]

                Ind_trunc_uniq_in_trunc = (Ind_trunc_uniq * In_trunc)[Pos_trunc_ind]

                if (Types_trunc == 1).any() and (Types_trunc == 2).any():

                    Si_count_O = compute_bonds(Pos_trunc,Types_trunc,threshold_Si=threshold_Si,threshold_O=threshold_O,threshold_H=threshold_H,do_count_type_3=True)[1]
                    O_count_Si = compute_bonds(Pos_trunc,Types_trunc,threshold_Si=threshold_Si,threshold_O=threshold_O,threshold_H=threshold_H,do_count_type_3=False)[2]


                    Si_count_O = Si_count_O[Ind_trunc_uniq_in_trunc[(Types_trunc==1).astype("bool")]]
                    Si_index = Ind_trunc_uniq[:Num_at][Types==1]
                    Si_count_O_tot[Si_index] = Si_count_O
                    O_count_Si = O_count_Si[Ind_trunc_uniq_in_trunc[(Types_trunc==2).astype("bool")]]
                    O_index = Ind_trunc_uniq[:Num_at][((Types==2)).astype("bool")]
                    O_count_Si_tot[O_index] = O_count_Si


                    Pos_Si = Pos_trunc_uniq[Types_trunc_uniq==1]
                    Pos_O = Pos_trunc_uniq[((Types_trunc_uniq==2) + (Types_trunc_uniq==3)).astype("bool")]
                    Dist_Si_O = sd.cdist(Pos_Si,Pos_O)
                    # Dist_Si_O = sd.cdist(Pos_Si,Pos_Si)
                    Dist_Si_O = Dist_Si_O[:Num_Si]
                    Dist_Si_O = Dist_Si_O + (Dist_Si_O==0)*100
                    Dist_Si_O = Dist_Si_O[Dist_Si_O<rdf_max].ravel()
                    Dist_list.append(Dist_Si_O)

    Si_count_O_tot = Si_count_O_tot[:Num_Si]
    O_count_Si_tot = O_count_Si_tot[:Num_O]

    return Dist_list, Si_count_O_tot, O_count_Si_tot



def plot_syst(Pos,Types,Cycles=None,L_cycles=None):

    Bonds, Si_count_O, O_count_Si, O_count_H, H_count_O = compute_bonds(Pos,Types)

    plotter = pv.Plotter()
    plotter.add_axes()



    sp = pv.Sphere(radius=0.6)
    Colors_Si = "dodgerblue"
    data = pv.PolyData(Pos[Types==1])
    pc = data.glyph(scale=False,geom=sp,orient=False)
    plotter.add_mesh(pc,opacity=0.5,pbr=True,roughness=.5,metallic=.2,color=Colors_Si)

    sp = pv.Sphere(radius=0.4)
    Colors_O = "red"
    data = pv.PolyData(Pos[Types==2])
    pc = data.glyph(scale=False,geom=sp,orient=False)
    plotter.add_mesh(pc,opacity=0.5,pbr=True,roughness=.5,metallic=.2,color=Colors_O)


    sp = pv.Sphere(radius=0.2)
    Colors_H = "gray"
    data = pv.PolyData(Pos[Types==3])
    pc = data.glyph(scale=False,geom=sp,orient=False)
    plotter.add_mesh(pc,opacity=0.5,pbr=True,roughness=.5,metallic=.2,color=Colors_H)


    N_Si,N_O = np.shape(Bonds)
    Indices = (np.arange(0,N_Si).reshape(N_Si,1,1)*np.array([1,0]) + np.arange(0,N_O).reshape((1,N_O,1))*np.array([0,1])).reshape((N_Si*N_O,2))

    Pos_Si = Pos[Types==1]
    Pos_O = Pos[Types==2]

    Indices = Indices[Bonds.ravel()!=0]

    tubes = [pv.Tube(Pos_Si[i_si],Pos_O[i_o],n_sides=5,radius=0.2) for i_si,i_o in Indices]
    mesh = tubes[0].merge(tubes[1:])
    plotter.add_mesh(mesh,opacity=0.3,pbr=True,roughness=.5,metallic=.2,color="blue")
    if not type(Cycles) is type(None) and type(L_cycles) is type(None):
        Pos_Si = Pos[Types==1]
        print(len(Pos_Si))
        tubes = []


        for cycle in Cycles:
            cycle2 = np.roll(cycle,1)

            for cycle_1,cycle_2 in zip(cycle,cycle2):
                tubes.append(pv.Tube(Pos_Si[cycle_1],Pos_Si[cycle_2],n_sides=5,radius=0.5))

        mesh = tubes[0].merge(tubes[1:])
        plotter.add_mesh(mesh,opacity=0.5,pbr=True,roughness=.5,metallic=.8,color="red")

    elif not type(Cycles) is type(None) and not type(L_cycles) is type(None):
        Pos_Si = Pos[Types==1]
        tubes = []

        for cycle in Cycles:
            cycle2 = np.roll(cycle,1)
            for cycle_1,cycle_2 in zip(cycle,cycle2):
                tubes.append(pv.Tube(Pos_Si[cycle_1],Pos_Si[cycle_2],n_sides=5,radius=0.5))

        mesh = tubes[0].merge(tubes[1:])
        plotter.add_mesh(mesh,opacity=0.5,pbr=True,roughness=.5,metallic=.8,color="red")



    plotter.show()



def analyze_plot_syst(Pos,Types,periodic=False,Lims=[],draw_limit=5,compute_limit=7,Cycles=None,L_cycles=None):

    def slide(value):
        if periodic:
            Pos_cut_l = (Pos[:,2] >= value-compute_limit) * (Pos[:,2] < value+compute_limit) + ((Pos[:,2] + Lims[2][1]) >= value-compute_limit) * ((Pos[:,2] + Lims[2][1]) < value+compute_limit) + ((Pos[:,2] - Lims[2][1]) > value-compute_limit) * ((Pos[:,2] - Lims[2][1]) < value+compute_limit)
            Pos_cut_l = Pos_cut_l > 0
        else:
            Pos_cut_l = (Pos[:,2] > value-compute_limit) * (Pos[:,2] < value+compute_limit)

        Pos_cutted_l = Pos[Pos_cut_l]
        Types_cutted_l = Types[Pos_cut_l]
        Bonds, Si_count_O, O_count_Si, O_count_H, H_count_O = compute_bonds(Pos_cutted_l,Types_cutted_l,periodic=periodic,Lims=Lims)
        Pos_cut = (Pos_cutted_l[:,2] > value-draw_limit) * (Pos_cutted_l[:,2] < value+draw_limit)

        #Remove outside
        Pos_cutted = Pos_cutted_l[Pos_cut]
        Types_cutted = Types_cutted_l[Pos_cut]
        select_Si = Pos_cut[Types_cutted_l==1]
        select_O = Pos_cut[((Types_cutted_l==2) + (Types_cutted_l==3)).astype("bool")]


        Si_count_O = Si_count_O[select_Si]
        O_count_Si = O_count_Si[select_O]
        if O_count_H.any(): O_count_H = O_count_H[select_O]




        sp = pv.Sphere(radius=0.6)
        Colors_Si = ["white","powderblue","lightsteelblue","dodgerblue","blue","navy","black"]
        for number_bonds in range(0,7):
            op = 1.0
            if number_bonds == 4: op = 0.05
            Si_bonds = (Si_count_O==number_bonds)
            if Si_bonds.any():
                data = pv.PolyData(Pos_cutted[Types_cutted==1][Si_bonds])
            else: data = pv.PolyData()
            pc = data.glyph(scale=False,geom=sp,orient=False)
            plotter.add_mesh(pc,opacity=op,pbr=True,roughness=.5,metallic=.2,color=Colors_Si[number_bonds],name="Si_{}".format(number_bonds))

        sp = pv.Sphere(radius=0.4)
        Colors_O = ["white","orange","red","darkred","black"]
        for number_bonds in range(0,5):
            op = 1.0
            if number_bonds == 2: op = 0.05
            if O_count_H.any():
                O_bonds = ((O_count_Si + O_count_H)==number_bonds)
            else:
                O_bonds = (O_count_Si==number_bonds)

            if O_bonds.any():
                data = pv.PolyData(Pos_cutted[((Types_cutted==2)+(Types_cutted==3)).astype("bool")][O_bonds])
            else: data = pv.PolyData()
            pc = data.glyph(scale=False,geom=sp,orient=False)
            plotter.add_mesh(pc,opacity=op,pbr=True,roughness=.5,metallic=.2,color=Colors_O[number_bonds],name="O_{}".format(number_bonds))


        sp = pv.Sphere(radius=0.2)
        Colors_H = ["white","gray","black"]
        data = pv.PolyData(Pos_cutted[Types_cutted==4])
        pc = data.glyph(scale=False,geom=sp,orient=False)
        plotter.add_mesh(pc,opacity=0.05,pbr=True,roughness=.5,metallic=.2,color="gray",name="H")

        if periodic:
            Bonds, Si_count_O, O_count_Si, O_count_H, H_count_O = compute_bonds(Pos_cutted_l,Types_cutted_l,periodic=False,Lims=Lims)
        Bonds = Bonds[select_Si]
        Bonds = Bonds[:,select_O]

            #Recompute bonds without periodicity otherwise bonds crossing
        N_Si,N_O = np.shape(Bonds)
        Indices = (np.arange(0,N_Si).reshape(N_Si,1,1)*np.array([1,0]) + np.arange(0,N_O).reshape((1,N_O,1))*np.array([0,1])).reshape((N_Si*N_O,2))

        Pos_Si = Pos_cutted[Types_cutted==1]
        Pos_O = Pos_cutted[((Types_cutted==2) + (Types_cutted==3)).astype("bool")]

        Indices = Indices[Bonds.ravel()!=0]
        tubes = pv.MultiBlock()
        for i_si,i_o in Indices:
            t = pv.Tube(Pos_Si[i_si],Pos_O[i_o],n_sides=5,radius=0.2)
            tubes.append(t)

        mesh = tubes.combine()
        plotter.add_mesh(mesh,opacity=0.1,pbr=True,roughness=.5,metallic=.2,color="blue",name="Bonds")



        if not type(Cycles) is type(None) and not type(L_cycles) is type(None):
            Pos_Si_l = Pos_cutted_l[Types_cutted_l==1]
            Ind_Si_cut = Pos_cut_l[Types==1]
            Ind_sum = np.cumsum(Ind_Si_cut)-1
            tubes_cycles = [pv.MultiBlock() for k in range(np.max(L_cycles)+1)]

            for cycle,l_cycle in zip(Cycles,L_cycles):
                cycle2 = np.roll(cycle,1)
                for cycle_1,cycle_2 in zip(cycle,cycle2):
                    if Ind_Si_cut[cycle_1] and Ind_Si_cut[cycle_2]:

                        # print(cycle_1,Ind_sum[cycle_1])
                        cycle_1 = Ind_sum[cycle_1]
                        cycle_2 = Ind_sum[cycle_2]
                        tubes_cycles[l_cycle].append(pv.Tube(Pos_Si_l[cycle_1],Pos_Si_l[cycle_2],n_sides=5,radius=0.5))

            colors = ["","","white","gray","green","red","pink","blue","purple","yellow","black","orange","magenta"]
            if len(L_cycles) > len(colors):
                colors = colors + colors[2:]

            for multblock,color,n in zip(tubes_cycles,colors,np.arange(len(tubes_cycles))):
                if multblock.n_blocks:
                    mesh = multblock.combine()
                    if n==6: opacity = 0.2
                    else: opacity = 0.3
                    plotter.add_mesh(mesh,opacity=opacity,pbr=True,roughness=.5,metallic=.8,color=color,name="cycles_{}".format(n))
                else: plotter.remove_actor("cycles_{}".format(n))


        plotter.add_point_labels(Pos_Si_l,np.arange(len(Pos_Si_l)),always_visible=True)



    z_min, z_max = np.min(Pos[:,2]), np.max(Pos[:,2])
    # print(np.shape(O_count_H))
    pv.global_theme.allow_empty_mesh = True
    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.add_slider_widget(slide, [z_min,z_max],value=(z_max+z_min)/2,title="Pos", fmt="%3.3e")

    plotter.show()




# def compute_analysis(Pos,Types,hist_Dens,hist_Si,hist_O,hist_H,threshold_Si=2,threshold_O=2,threshold_H=1.3,periodic=False,Lims=[],rdf_max=5):
def compute_analysis(Pos,Types,hist_Dens,hist_Si,hist_O,threshold_Si=2,threshold_O=2,threshold_H=1.3,periodic=False,Lims=[],rdf_max=5):

    Dist_list, Si_count_O, O_count_Si = compute_hist_neighbors(Pos,Types,threshold_Si=threshold_Si,threshold_O=threshold_O,threshold_H=threshold_H,periodic=periodic,Lims=Lims,rdf_max=rdf_max)
    #
    Dist_list = [k for j in Dist_list for k in j]

    num_inf_15 = np.sum(np.array(Dist_list) < 1.5)
    num_inf_155 = np.sum(np.array(Dist_list) < 1.55)
    num_si_3 = np.sum(Si_count_O==3)
    num_si_5 = np.sum(Si_count_O==5)
    num_si_6 = np.sum(Si_count_O==6)
    num_o_1 = np.sum(O_count_Si==1)
    num_o_3 = np.sum(O_count_Si==3)

    print("Bonds < 1.5 A : {}".format(num_inf_15))
    print("Bonds < 1.55 A : {}".format(num_inf_155))
    print("3 Bonds Si : {}".format(num_si_3))
    print("5 Bonds Si : {}".format(num_si_5))
    print("6 Bonds Si : {}".format(num_si_6))
    print("1 Bonds O : {}".format(num_o_1))
    print("3 Bonds O : {}".format(num_o_3))



    hist_Dens = np.histogram(Dist_list,hist_Dens[1])[0]
    # hist_Dens = np.histogram(Dens_trunc,hist_Dens[1])[0]
    hist_Si = np.histogram(Si_count_O,hist_Si[1])[0]
    hist_O = np.histogram(O_count_Si,hist_O[1])[0]
    # hist_H = np.histogram(O_count_H,hist_H[1])[0]


    # Counts = [np.sum(Si_count_O), np.sum(O_count_Si), np.sum(O_count_H)]
    Counts = [np.sum(Si_count_O), np.sum(O_count_Si)]
    print(Counts)

    return [hist_Dens,hist_Si,hist_O],Counts
    return [hist_Dens,hist_Si,hist_O,hist_H],Counts



# def plot_analysis(Counts_Hists,Counts,hist_Dens,hist_Si,hist_O,hist_H):
def plot_analysis(Counts_Hists,Counts,hist_Dens,hist_Si,hist_O):

    Counts_Hist_Dens = Counts_Hists[0]
    for count, rect in zip(Counts_Hist_Dens,hist_Dens[2].patches):
        rect.set_height(count)

    # Si_count_O, O_count_Si, O_count_H = Counts
    Si_count_O, O_count_Si = Counts


    Counts_Hist_Si = Counts_Hists[1]
    for count, rect in zip(Counts_Hist_Si,hist_Si[2].patches):
        rect.set_height(count)
    hist_Si[2].set_label("Si-O Bonds {}".format(Counts[0]))

    # if Counts[2]>0:
    if 0:
        Counts_Hist_O = Counts_Hists[2]
        for count, rect in zip(Counts_Hist_O,hist_O[2].patches):
            rect.set_height(count)

        Counts_Hist_H = Counts_Hists[3]
        Counts_Hist_H[0]=0
        for count, rect in zip(Counts_Hist_H,hist_H[2].patches):
            rect.set_height(count)

        hist_O[2].set_label("O-Si Bonds {}".format(Counts[1]))
        hist_H[2].set_label("O-H Bonds {}".format(Counts[2]))

    else:
        Counts_Hist_O = Counts_Hists[2]
        for count, rect in zip(Counts_Hist_O,hist_O[2].patches):
            rect.set_height(count)
        hist_O[2].set_label("O-Si Bonds {}".format(Counts[1]))


def analyze_mult(list_Tstep,list_Pos,list_Types,threshold_Si=2,threshold_O=2,threshold_H=1.3,rdf_max=5,periodic=False,Lims=[],anim=False,save=False):
    fig,ax = plt.subplots()
    if save:
        fig,ax = plt.subplots(figsize=(8,6),dpi=200)
        plt.rcParams.update({'font.size': 15})
        plt.rcParams['svg.fonttype'] = 'none'

    plt.axis("off")
    if not save:
        fig.subplots_adjust(bottom=0.25)
        ax_slider = fig.add_axes([0.25,0.1,0.65,0.03])
        slider = Slider(ax=ax_slider,label="Timestep",valmin=list_Tstep[0],valmax=list_Tstep[-1],valinit=list_Tstep[0],valfmt="%d",valstep=list_Tstep)
    num_O = np.sum(list_Types[0] == 2)
    num_Si = np.sum(list_Types[0] == 1)

    purple = np.array([96,25,255])/255
    dark_purple = np.array([56,20,180])/255

    plt.subplot(2,1,1)
    hist_Dens = plt.hist(np.array([]),bins=100,range=(0,5),color=purple,edgecolor=dark_purple,linewidth=1,label="A")
    plt.title("RDF Si-O",color=dark_purple)
    plt.ylabel("Number",color=dark_purple)
    plt.xlabel("Distance (A)",color=dark_purple)
    plt.xticks(color=purple)
    plt.yticks(color=purple)
    plt.xlim(0,5)
    plt.ylim(0,num_O*2)

    plt.subplot(2,2,3)
    hist_Si = plt.hist(np.array([]),bins=12,range=(0,6),color=purple,edgecolor=dark_purple,linewidth=1,label="A")
    plt.title("Number of Bonds for Si",color=dark_purple)
    plt.xlabel("Number of Bonds",color=dark_purple)
    plt.ylabel("Number of Si",color=dark_purple)
    plt.xticks([k+0.25 for k in range(7)],[k for k in range(7)],color=purple)
    plt.yticks(color=purple)
    plt.ylim(0,num_Si*1.2)


    plt.subplot(2,2,4)
    hist_O = plt.hist(np.array([]),bins=12,range=(0,6),color=purple,edgecolor=dark_purple,linewidth=1,label="A")
    # hist_H = plt.hist(np.array([]),bins=12,range=(0,6),color="blue",edgecolor="darkblue",linewidth=1,label="A")
    plt.xticks([k+0.25 for k in range(7)],[k for k in range(7)],color=purple)
    plt.yticks(color=purple)
    plt.title("Number of Bonds for O",color=dark_purple)
    plt.xlabel("Number of Bonds",color=dark_purple)
    plt.ylabel("Number of O",color=dark_purple)
    plt.ylim(0,num_O*1.2)

    if anim:
        plt.show(block=False)



    list_Counts_Hists,list_Counts = [], []
    for tstep in range(len(list_Tstep)):
        Hist_Counts,Counts = compute_analysis(list_Pos[tstep],list_Types[tstep],hist_Dens,hist_Si,hist_O,threshold_Si=threshold_Si,threshold_O=threshold_O,threshold_H=threshold_H,periodic=periodic,Lims=Lims,rdf_max=rdf_max)
        # Hist_Counts,Counts = compute_analysis(list_Pos[tstep],list_Types[tstep],hist_Dens,hist_Si,hist_O,hist_H,threshold_Si=threshold_Si,threshold_O=threshold_O,threshold_H=threshold_H,periodic=periodic,Lims=Lims,rdf_max=rdf_max)
        if anim:
            slider.set_val(list_Tstep[tstep])
            # plot_analysis(Hist_Counts,Counts,hist_Dens,hist_Si,hist_O,hist_H)
            plot_analysis(Hist_Counts,Counts,hist_Dens,hist_Si,hist_O)
            plt.pause(0.02)

        list_Counts_Hists.append(Hist_Counts)
        list_Counts.append(Counts)


    def update(val):
        index = list_Tstep.index(val)
        # plot_analysis(list_Counts_Hists[index],list_Counts[index],hist_Dens,hist_Si,hist_O,hist_H)
        plot_analysis(list_Counts_Hists[index],list_Counts[index],hist_Dens,hist_Si,hist_O)
        plt.draw()

    if not anim and not save:
        slider.on_changed(update)
        update(list_Tstep[0])
        plt.show()

    if save:
        update(list_Tstep[-1])
        plt.tight_layout()
        plt.savefig("analysis.svg")



def analyze_defects(Pos,Types,periodic=False,Lims=[],Cycles=None,L_cycles=None):

    D, Si_count_O, O_count_Si =  compute_hist_neighbors(Pos,Types,cube=30,threshold_Si=2,threshold_O=2,threshold_H=1.3,periodic=periodic,Lims=Lims,rdf_max=5)

    plotter = pv.Plotter()
    plotter.add_axes()

    sp = pv.Sphere(radius=0.6)
    Colors_Si = ["white","powderblue","lightsteelblue","dodgerblue","blue","navy","black"]
    for number_bonds in range(0,7):
        op = 1.0
        if number_bonds == 4: pass
        else:
            Si_bonds = (Si_count_O==number_bonds)
            if Si_bonds.any():
                data = pv.PolyData(Pos[Types==1][Si_bonds])
                pc = data.glyph(scale=False,geom=sp,orient=False)
                plotter.add_mesh(pc,opacity=op,pbr=True,roughness=.5,metallic=.2,color=Colors_Si[number_bonds],name="Si_{}".format(number_bonds))
            else: plotter.remove_actor("Si_{}".format(number_bonds))


    sp = pv.Sphere(radius=0.4)
    Colors_O = ["white","orange","red","darkred","black"]
    for number_bonds in range(0,5):
        op = 1.0
        if number_bonds==2: pass
        else:
            # if O_count_H.any():
            #     O_bonds = ((O_count_Si + O_count_H)==number_bonds)
            # else:
            O_bonds = (O_count_Si==number_bonds)
            # print(np.shape(O_count_Si),np.shape(Pos[((Types==2)+(Types==3)).astype("bool")]))

            if O_bonds.any():
                data = pv.PolyData(Pos[((Types==2)).astype("bool")][O_bonds])
                pc = data.glyph(scale=False,geom=sp,orient=False)
                plotter.add_mesh(pc,opacity=op,pbr=True,roughness=.5,metallic=.2,color=Colors_O[number_bonds],name="O_{}".format(number_bonds))
            else: plotter.remove_actor("O_{}".format(number_bonds))

    num_si_3 = np.sum(Si_count_O==3)
    num_si_5 = np.sum(Si_count_O==5)
    num_si_6 = np.sum(Si_count_O==6)
    num_o_1 = np.sum(O_count_Si==1)
    num_o_3 = np.sum(O_count_Si==3)


    print("3 Bonds Si : {}".format(num_si_3))
    print("5 Bonds Si : {}".format(num_si_5))
    print("6 Bonds Si : {}".format(num_si_6))
    print("1 Bonds O : {}".format(num_o_1))
    print("3 Bonds O : {}".format(num_o_3))
    # sp = pv.Sphere(radius=0.2)
    # Colors_H = ["white","gray","black"]
    # data = pv.PolyData(Pos_cutted[Types_cutted==4])
    # pc = data.glyph(scale=False,geom=sp,orient=False)
    # plotter.add_mesh(pc,opacity=0.05,pbr=True,roughness=.5,metallic=.2,color="gray",name="H")

    # if periodic:
    #     Bonds, Si_count_O, O_count_Si, O_count_H, H_count_O = compute_bonds(Pos_cutted_l,Types_cutted_l,periodic=False,Lims=Lims)
    # Bonds = Bonds[select_Si]
    # Bonds = Bonds[:,select_O]
    #
    #     #Recompute bonds without periodicity otherwise bonds crossing
    # N_Si,N_O = np.shape(Bonds)
    # Indices = (np.arange(0,N_Si).reshape(N_Si,1,1)*np.array([1,0]) + np.arange(0,N_O).reshape((1,N_O,1))*np.array([0,1])).reshape((N_Si*N_O,2))
    #
    # Pos_Si = Pos_cutted[Types_cutted==1]
    # Pos_O = Pos_cutted[((Types_cutted==2) + (Types_cutted==3)).astype("bool")]
    #
    # Indices = Indices[Bonds.ravel()!=0]
    # tubes = pv.MultiBlock()
    # for i_si,i_o in Indices:
    #     t = pv.Tube(Pos_Si[i_si],Pos_O[i_o],n_sides=5,radius=0.2)
    #     tubes.append(t)
    #
    # mesh = tubes.combine()
    # plotter.add_mesh(mesh,opacity=0.1,pbr=True,roughness=.5,metallic=.2,color="blue",name="Bonds")



    if not type(Cycles) is type(None) and not type(L_cycles) is type(None):
        Pos_Si_l = Pos[Types_cutted_l==1]
        Ind_Si_cut = Pos[Types==1]
        Ind_sum = np.cumsum(Ind_Si_cut)-1
        tubes_cycles = [pv.MultiBlock() for k in range(np.max(L_cycles)+1)]

        for cycle,l_cycle in zip(Cycles,L_cycles):
            cycle2 = np.roll(cycle,1)
            for cycle_1,cycle_2 in zip(cycle,cycle2):
                if Ind_Si_cut[cycle_1] and Ind_Si_cut[cycle_2]:

                    # print(cycle_1,Ind_sum[cycle_1])
                    cycle_1 = Ind_sum[cycle_1]
                    cycle_2 = Ind_sum[cycle_2]
                    tubes_cycles[l_cycle].append(pv.Tube(Pos_Si_l[cycle_1],Pos_Si_l[cycle_2],n_sides=5,radius=0.5))

        colors = ["","","white","gray","green","red","pink","blue","purple","yellow","black","orange","magenta"]
        if len(L_cycles) > len(colors):
            colors = colors + colors[2:]

        for multblock,color,n in zip(tubes_cycles,colors,np.arange(len(tubes_cycles))):
            if multblock.n_blocks:
                mesh = multblock.combine()
                if n==6: opacity = 0.2
                else: opacity = 0.3
                plotter.add_mesh(mesh,opacity=opacity,pbr=True,roughness=.5,metallic=.8,color=color,name="cycles_{}".format(n))
            else: plotter.remove_actor("cycles_{}".format(n))


    nx,ny,nz = 40,20,150
    D = 40
    x_i = np.linspace(0,80,nx,endpoint=True)
    y_i = np.linspace(0,40,ny,endpoint=True)
    z_i = np.linspace(0,300,nz,endpoint=True)

    X,Y,Z = np.meshgrid(x_i,y_i,z_i)


    Pos_surf = np.array([X,Y,Z]).transpose((1,2,3,0))

    Pos_transfo,slide_z,mean = transfo(Pos_surf,slide_z=0,D=D,rota=1,do_periodic=False)
    x,y,z = Pos_transfo

    x_m,y_m,z_m = np.mean(Pos,axis=0)
    x_m_t,y_m_t,z_m_t = np.mean(np.mean(np.mean(Pos_transfo,axis=1),axis=1),axis=1)
    x = x - x_m_t + x_m
    y = y - y_m_t + y_m

    grid = pv.StructuredGrid(x,y,z)
    # plotter.add_mesh(grid,name="a",opacity=0.3,pbr=True,roughness=0.6,metallic=0.8,color=[56,0,180])

    plotter.show()



def quicksurf(Pos,N_list=[20,20,80],alpha=1):
    Lx,Ly,Lz = np.max(Pos,axis=0)
    lx,ly,lz = np.min(Pos,axis=0)

    Nx,Ny,Nz = N_list

    cube = np.zeros((Nx,Ny,Nz))
    x = np.linspace(lx,Lx,Nx,endpoint=True)
    y = np.linspace(ly,Ly,Ny,endpoint=True)
    z = np.linspace(lz,Lz,Nz,endpoint=True)

    dx = x[1]-x[0]
    dy = y[1]-y[0]
    dz = z[1]-z[0]

    cube_coords = x.reshape((1,Nx,1,1))*np.array([1.,0,0]).reshape((3,1,1,1)) + \
        y.reshape((1,1,Ny,1))*np.array([0,1.,0]).reshape((3,1,1,1)) + \
        z.reshape((1,1,1,Nz))*np.array([0,0,1.]).reshape((3,1,1,1))

    grid = pv.RectilinearGrid(x,y,z)

    def slider_precision(value):
        for x in range(Nx):
            for y in range(Ny):
                for z in range(Nz):
                    cube_pos = cube_coords[:,x,y,z].reshape((1,3))

                    Pos_trunc_x = (Pos[:,0] >= (x * dx - value)) * (Pos[:,0] < (x * dx + value))
                    Pos_trunc_y = (Pos[:,1] >= (y * dy - value)) * (Pos[:,1] < (y * dy + value))
                    Pos_trunc_z = (Pos[:,2] >= (z * dz - value)) * (Pos[:,2] < (z * dz + value))
                    Ind_trunc = Pos_trunc_x * Pos_trunc_y * Pos_trunc_z
                    Pos_trunc = Pos[Ind_trunc]
                    Dist = sd.cdist(cube_pos, Pos_trunc)

                    cube[x,y,z] = np.sum(np.exp(-Dist**2/alpha))



        cube_density = cube / np.sum(cube)

        array_sort = np.argsort(cube_density,axis=None)[::-1]
        cube_sorted = cube_density.flatten()[array_sort]
        cube_values_sorted = np.cumsum(cube_sorted)

        #Find how to unsort the array. There should be a more efficient way to do this
        indexes = np.arange(len(array_sort),dtype=int)
        array_unsort = np.zeros(len(array_sort),dtype=int)

        for k in range(len(array_sort)):
            array_unsort[array_sort[k]] = indexes[k]
        cube_values = cube_values_sorted[array_unsort]


        contour= grid.contour(isosurfaces=[0.8],scalars=cube_values)
        plotter.add_mesh(contour,name="contour",opacity=1.0,pbr=True,roughness=.5,metallic=.2,color="red")


    plotter = pv.Plotter()

    plotter.add_slider_widget(slider_precision, [0.5,8],value=1,title="Precision", fmt="%1.1f")
    plotter.show(full_screen=False)


if __name__=="__main__":
    # file = "dump_last_oh.lammpstrj"
    # file = "dummp_trimmed.lammpstrj"
    # file = "dummps_snad_last.lammpstrj"
    file = "dummp_700_last.lammpstrj"
    # file = "dummp_twisted_long.lammpstrj"
    # file = "quartz_dupl.data"
    # file = "dummps_snad_last.lammpstrj"
    # file = "dummps_round_2_last.lammpstrj"
    # file = "dummps_round_3_last.lammpstrj"
    # file = "dummps_long_last.lammpstrj"
    # file = "dummp_trimmed_long_0K.lammpstrj"
    # file = "dummp_trimmed_0K_last.lammpstrj"

    list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS = read_dump(file,unscale=True)
    # list_BOX,list_ATOMS = read_data(file,do_scale=False)
    list_TSTEP=[0]

    list_Pos = list_ATOMS[:,:,2:]
    list_Types = list_ATOMS[:,:,1]
    Pos = list_ATOMS[-1][:,2:]
    Types = list_ATOMS[-1][:,1]



    # analyze(Pos,Types,periodic=True,Lims=list_BOX[-1])
    # plot_syst(Pos,Types,Lims=list_BOX[-1])
    analyze_mult(list_TSTEP,list_Pos,list_Types,periodic=False,Lims=list_BOX[-1],save=True)
    # quicksurf(Pos,N_list=[40,40,40],alpha=0.5)
    # analyze_defects(Pos,Types,periodic=True,Lims=list_BOX[-1])
