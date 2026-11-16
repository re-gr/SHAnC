"""
Analysis and system visualisation utilities for SHAnC.

This module provides bond detection, coordination analysis, radial
distribution function (RDF) computation, structural validation, and
visualization helpers for oxide and metal systems.

Features:
    - compute bond graphs from atom positions and types
    - evaluate coordination and bond statistics
    - plot histograms, RDFs, and density maps
    - create 3D visualizations with PyVista

Dependencies:
    numpy, matplotlib, pyvista, scipy
"""

import numpy as np
import shanc
import scipy.spatial.distance as sd
from shanc.read_write import *
import matplotlib.pyplot as plt
import pyvista as pv
from shanc.helix_object import *
import threading as th


# Colours extracted from Jmol
# https://jmol.sourceforge.net/jscolors/
ELEMENT_COLOURS = {
    "H":	[255,255,255], "He":	[217,255,255], "Li":	[204,128,255], "Be":	[194,255,0],
    "B":	[255,181,181],  "C":	[72,72,72],     "N":	[48,80,248]	,   "O":	[255,13,13]	,
    "F":	[144,224,80],  "Ne":	[179,227,245], "Na":	[171,92,242],  "Mg":	[138,255,0]	,
    "Al":	[191,166,166], "Si":	[240,200,160],  "P":	[255,128,0]	,   "S":	[255,255,48],
    "Cl":	[31,240,31]	,  "Ar":	[128,209,227],  "K":	[143,64,212],  "Ca":	[61,255,0]	,
    "Sc":	[230,230,230], "Ti":	[191,194,199],  "V":	[166,166,171], "Cr":	[138,153,199],
    "Mn":	[156,122,199], "Fe":	[224,102,51],  "Co":	[240,144,160], "Ni":	[80,208,80]	,
    "Cu":	[200,128,51],  "Zn":	[125,128,176], "Ga":	[194,143,143], "Ge":	[102,143,143],
    "As":	[189,128,227], "Se":	[255,161,0]	,  "Br":	[166,41,41]	,  "Kr":	[92,184,209],
    "Rb":	[112,46,176],  "Sr":	[0,255,0]	,   "Y":	[148,255,255], "Zr":	[148,224,224],
    "Nb":	[115,194,201], "Mo":	[84,181,181],  "Tc":	[59,158,158],  "Ru":	[36,143,143],
    "Rh":	[10,125,140],  "Pd":	[0,105,133]	,  "Ag":	[192,192,192], "Cd":	[255,217,143],
    "In":	[166,117,115], "Sn":	[102,128,128], "Sb":	[158,99,181],  "Te":	[212,122,0]	,
    "I":	[148,0,148]	,  "Xe":	[66,158,176],  "Cs":	[87,23,143],   "Ba":	[0,201,0],
    "La":	[112,212,255], "Ce":	[255,255,199], "Pr":	[217,255,199], "Nd":	[199,255,199],
    "Pm":	[163,255,199], "Sm":	[143,255,199], "Eu":	[97,255,199],  "Gd":	[69,255,199],
    "Tb":	[48,255,199],  "Dy":	[31,255,199],  "Ho":	[0,255,156]	,  "Er":	[0,230,117]	,
    "Tm":	[0,212,82]	,  "Yb":	[0,191,56]	,  "Lu":	[0,171,36]	,  "Hf":	[77,194,255],
    "Ta":	[77,166,255],   "W":	[33,148,214],  "Re":	[38,125,171],  "Os":	[38,102,150],
    "Ir":	[23,84,135]	,  "Pt":	[208,208,224], "Au":	[255,209,35],  "Hg":	[184,184,208],
    "Tl":	[166,84,77]	,  "Pb":	[87,89,97]	,  "Bi":	[158,79,181],  "Po":	[171,92,0]	,
    "At":	[117,79,69]	,  "Rn":	[66,130,150],  "Fr":	[66,0,102]	,  "Ra":	[0,125,0]	,
    "Ac":	[112,171,250], "Th":	[0,186,255]	,  "Pa":	[0,161,255]	,   "U":	[0,143,255]	,
    "Np":	[0,128,255]	,  "Pu":	[0,107,255]	,  "Am":	[84,92,242]	,  "Cm":	[120,92,227],
    "Bk":	[138,79,227],  "Cf":	[161,54,212],  "Es":	[179,31,212],  "Fm":	[179,31,186],
    "Md":	[179,13,166],  "No":	[189,13,135],  "Lr":	[199,0,102]	,  "Rf":	[204,0,89]	,
    "Db":	[209,0,79]	,  "Sg":	[217,0,69]	,  "Bh":	[224,0,56]	,  "Hs":	[230,0,46]	,
    "Mt":	[235,0,38]	,  "Ds":   [250,0,0],      "Rg":[250,0,0],         "Cn":[250,0,0],
    "Nh":[250,0,0],        "Fl":[250,0,0],         "Mc":[250,0,0],         "Lv":[250,0,0],
    "Ts":[250,0,0],        "Og":[250,0,0]
}

ELEMENT_RADIUS = {
    'H':0.83, 'He':0.87, 'Li':0.91, 'Be':0.94, 'B':0.97, 'C':0.99, 'N':1.01, 'O':1.04, 'F':1.06, 'Ne':1.07, 'Na':1.09, 'Mg':1.11, 'Al':1.12,
    'Si':1.14, 'P':1.15, 'S':1.16, 'Cl':1.18, 'Ar':1.19, 'K':1.2, 'Ca':1.21, 'Sc':1.22, 'Ti':1.23, 'V':1.24, 'Cr':1.25, 'Mn':1.26, 'Fe':1.26,
    'Co':1.27, 'Ni':1.28, 'Cu':1.41, 'Zn':1.41, 'Ga':1.3, 'Ge':1.31, 'As':1.31, 'Se':1.32, 'Br':1.33, 'Kr':1.33, 'Rb':1.34, 'Sr':1.34, 'Y':1.35,
    'Zr':1.35, 'Nb':1.36, 'Mo':1.36, 'Tc':1.37, 'Ru':1.37, 'Rh':1.38, 'Pd':1.38, 'Ag':1.39, 'Cd':1.39, 'In':1.4, 'Sn':1.4, 'Sb':1.4, 'Te':1.41,
    'I':1.41, 'Xe':1.42,'Cs':1.42, 'Ba':1.42, 'La':1.43, 'Ce':1.43, 'Pr':1.43, 'Nd':1.44, 'Pm':1.44, 'Sm':1.44, 'Eu':1.45, 'Gd':1.45, 'Tb':1.45,
    'Dy':1.46, 'Ho':1.46, 'Er':1.46, 'Tm':1.47, 'Yb':1.47, 'Lu':1.47, 'Hf':1.48, 'Ta':1.48, 'W':1.48, 'Re':1.48, 'Os':1.49, 'Ir':1.49, 'Pt':1.49,
    'Au':1.49, 'Hg':1.5, 'Tl':1.5, 'Pb':1.5, 'Bi':1.5, 'Po':1.51, 'At':1.51, 'Rn':1.51, 'Fr':1.51, 'Ra':1.52, 'Ac':1.52, 'Th':1.52, 'Pa':1.52,
    'U':1.52, 'Np':1.53, 'Pu':1.53, 'Am':1.53, 'Cm':1.53, 'Bk':1.53, 'Cf':1.54, 'Es':1.54, 'Fm':1.54, 'Md':1.54, 'No':1.54, 'Lr':1.55, 'Rf':1.55,
    'Db':1.55, 'Sg':1.55, 'Bh':1.55, 'Hs':1.56, 'Mt':1.56, 'Ds':1.6, 'Rg':1.6, 'Cn':1.6, 'Nh':1.6, 'Fl':1.6, 'Mc':1.6, 'Lv':1.6, 'Ts':1.6, 'Og':1.6
}

_PURPLE = np.array([96, 25, 255]) / 255
_DARK_PURPLE = np.array([56, 20, 180]) / 255
_DARK_DARK_PURPLE = np.array([34, 10, 120]) / 255


ELEMENT_ACCEPTABLE_BONDS = {
    'H':1, 'Li':1, 'Be':2, 'B':3, 'C':4, 'N':3, 'O':2, 'F':1, 'Na':1, 'Mg':2, 'Al':3,
    'Si':4, 'P':3, 'S':2, 'Cl':1
}

#Basically 2 except for H
ELEMENTS_BONDS_THRESHOLDS = {
    'H': 1.3, 'He': 2.0, 'Li': 2.0, 'Be': 2.0, 'B': 2.0, 'C': 2.0, 'N': 2.0, 'O': 2.0, 'F': 2.0, 'Ne': 2.0, 'Na': 2.0, 'Mg': 2.0, 'Al': 2.0, 'Si': 2.0, 'P': 2.0, 'S': 2.0, 'Cl': 2.0, 'Ar': 2.0,
    'K': 3.0, 'Ca': 3.0, 'Sc': 3.0, 'Ti': 3.0, 'V': 3.0, 'Cr': 3.0, 'Mn': 3.0, 'Fe': 3.0, 'Co': 3.0, 'Ni': 3.0, 'Cu': 3.0, 'Zn': 3.0, 'Ga': 3.0, 'Ge': 3.0, 'As': 3.0, 'Se': 3.0, 'Br': 3.0, 'Kr': 3.0, 'Rb': 3.0, 'Sr': 3.0, 'Y': 3.0, 'Zr': 3.0, 'Nb': 3.0, 'Mo': 3.0, 'Tc': 3.0, 'Ru': 3.0, 'Rh': 3.0, 'Pd': 3.0, 'Ag': 3.0, 'Cd': 3.0, 'In': 3.0, 'Sn': 3.0, 'Sb': 3.0, 'Te': 3.0, 'I': 3.0, 'Xe': 3.0, 'Cs': 3.0, 'Ba': 3.0, 'La': 3.0, 'Ce': 3.0, 'Pr': 3.0, 'Nd': 3.0, 'Pm': 3.0, 'Sm': 3.0, 'Eu': 3.0, 'Gd': 3.0, 'Tb': 3.0, 'Dy': 3.0, 'Ho': 3.0, 'Er': 3.0, 'Tm': 3.0, 'Yb': 3.0, 'Lu': 3.0, 'Hf': 3.0, 'Ta': 3.0, 'W': 3.0, 'Re': 3.0, 'Os': 3.0, 'Ir': 3.0, 'Pt': 3.0, 'Au': 3.0, 'Hg': 3.0, 'Tl': 3.0, 'Pb': 3.0, 'Bi': 3.0, 'Po': 3.0, 'At': 3.0, 'Rn': 3.0, 'Fr': 3.0, 'Ra': 3.0, 'Ac': 3.0, 'Th': 3.0, 'Pa': 3.0, 'U': 3.0, 'Np': 3.0, 'Pu': 3.0, 'Am': 3.0, 'Cm': 3.0, 'Bk': 3.0, 'Cf': 3.0, 'Es': 3.0, 'Fm': 3.0, 'Md': 3.0, 'No': 3.0, 'Lr': 3.0, 'Rf': 3.0, 'Db': 3.0, 'Sg': 3.0, 'Bh': 3.0, 'Hs': 3.0, 'Mt': 3.0, 'Ds': 3.0, 'Rg': 3.0, 'Cn': 3.0, 'Nh': 3.0, 'Fl': 3.0, 'Mc': 3.0, 'Lv': 3.0, 'Ts': 3.0, 'Og': 3.0
}



_FIGSIZE = (12, 8)
_FONT = 20

# ---------------------------------------------------------------------------
# Bond computation
# ---------------------------------------------------------------------------


def _compute_bonds(Pos, Types, type_map, threshold_map, threshold_mult = 1.1):
    """
    _compute_bonds(Pos, Types, type_map, threshold_map, threshold_mult = 1.1)

    Compute the bonds of a system

    This version must only be used for small systems; it builds the full
    distance matrix and is memory-intensive for large systems.


    Parameters
    ----------
        Pos: ndarray
            The position of the atoms
        Types : ndarray
            The types of the atoms
        type_map : dict
            The dictionary that links types to atoms
        threshold_map : dict
            The dictionary that links atoms to the threshold to find bonds
        threshold_mult : float, optional
            Multiplier for all the thresholds. By default 1.1

    Returns
    -------
        Bonds
            The Bonds of the system as a list of pairs
    """

    max_type = int(np.max(Types))
    Bonds = np.zeros((len(Pos),len(Pos)))

    Thresholds = np.zeros((len(Pos)))
    for type_ in range(max_type):
        Thresholds = Thresholds + threshold_map[type_map[type_+1]] * (Types==type_+1).astype("float") * threshold_mult


    for type_ in range(max_type):
        mask_type = (Types==type_+1)
        Pos_type = Pos[mask_type]
        threshold_type = threshold_map[type_map[type_+1]] * threshold_mult
        Threshold_type = Thresholds + (Thresholds < threshold_type) * (threshold_type - Thresholds)
        Threshold_type.reshape((1,len(Threshold_type)))

        Dist = sd.cdist(Pos_type,Pos)
        # print(np.shape(Dist))
        #Remove count with itself
        Dist = Dist + 100 * (Dist <= 1e-8)
        mask_bonds = (Dist < Threshold_type)

        Bonds[mask_type] = Bonds[mask_type] + mask_bonds

    Bonds = (Bonds>=1)


    return Bonds



def compute_bonds_system(helix,cube=30,d_max=5):
    """
    compute_bonds_system(helix,cube=30,d_max=5)

    Compute bond counts and RDF distances by slicing the system into sub-cubes.

    Parameters
    ----------
        helix : Helix
            The helix on which to compute the bonds
        cube : float, optional
            the length parameter of the cubes used to divide the system. By default 30 A
        d_max : float, optional
            the length by which each cube is enlarged to compute bonds with other cubes. By default 5 A

    Returns
    -------
        Bonds
            The Bonds of the system as a list of pairs
    """
    N_th = shanc.nprocs

    Pos = helix.Pos
    Types = helix.Types
    Lims = helix.Lims
    type_map = helix.type_map
    threshold_map = helix.threshold_map
    periodic = helix.periodic

    if threshold_map == {}:
        threshold_map = ELEMENTS_BONDS_THRESHOLDS

    Lx, Ly, Lz = np.max(Pos, axis=0)
    lx, ly, lz = np.min(Pos, axis=0)

    if len(helix.Bonds_analysis):
        return helix.Bonds_analysis

    # Use the system bounds from Pos unless periodic limits are explicitly provided.
    if periodic:
        if len(Lims) == 0:
            print("No limits provided – running as non-periodic.")
            periodic = False
        else:
            lz, Lz = Lims[2]

    # Divide the cell into cubic subregions for scalable neighbor counting.
    Nx = int((Lx - lx) // cube + 1)
    Ny = int((Ly - ly) // cube + 1)
    Nz = int((Lz - lz) // cube + 1)

    Pos_added   = np.copy(Pos)
    Types_added = np.copy(Types)

    if periodic:
        Pos_add_z    = Pos[:, 2] > (Lz - d_max)
        Pos_remove_z = Pos[:, 2] < (lz + d_max)

        Pos_added = np.vstack([
            Pos_added,
            Pos[Pos_add_z]    - [0, 0, Lz - lz],
            Pos[Pos_remove_z] + [0, 0, Lz - lz],
        ])
        Types_added = np.concatenate([
            Types_added,
            Types[Pos_add_z],
            Types[Pos_remove_z],
        ])


    Bonds = []
    Index = np.arange(len(Pos_added))

    if periodic:
        Index_small = np.arange(len(Pos))
        Index_add = np.concatenate([Index_small[Pos_add_z],Index_small[Pos_remove_z]])
        Index[len(Pos):] = Index_add

    pos_x = Pos_added[:,0]
    pos_y = Pos_added[:,1]
    pos_z = Pos_added[:,2]

    def evaluate_bonds(Bonds,Nz_range):
        """function to multithread"""
        for z in range(Nz_range[0],Nz_range[1]):
            select_z = (pos_z >= z*cube + lz - d_max) & (pos_z < (z+1)*cube + lz + d_max)
            for y in range(Ny):
                select_y = (pos_y >= y*cube + ly - d_max) & (pos_y < (y+1)*cube + ly + d_max)
                for x in range(Nx):
                    #Extract small cubes to compute the bonds
                    mask_p = select_z & select_y & (pos_x >= x*cube + lx - d_max) & (pos_x < (x+1)*cube + lx + d_max)

                    Pos_trunc   = Pos_added[mask_p]
                    Types_trunc = Types_added[mask_p]
                    if len(Types_trunc) == 0: continue

                    bonds = _compute_bonds(Pos_trunc, Types_trunc, type_map, threshold_map)
                    index_p = Index[mask_p]
                    n_index_p = len(index_p)
                    index_bonds_p = (index_p.reshape((n_index_p,1,1)) * np.array([[1,0]]) + index_p.reshape((1,n_index_p,1)) * np.array([[0,1]])).reshape((n_index_p*n_index_p,2))
                    index_bonds_p = index_bonds_p[bonds.ravel()!=0]

                    for index_b in index_bonds_p:
                        Bonds.append(index_b)



    Nz_per_thread = [Nz // N_th] * N_th
    for j in range(Nz % N_th):
        Nz_per_thread[j] += 1

    ranges, count = [], 0
    for n in Nz_per_thread:
        ranges.append([count, count + n])
        count += n

    threads = []
    for rng in ranges:
        if rng[0] != rng[1]:
            t = th.Thread(target=evaluate_bonds, args=(Bonds,rng))
            threads.append(t)
            t.start()
    for t in threads:
        t.join()


    Bonds = np.array(Bonds)
    helix.Bonds_analysis = Bonds

    return Bonds


def compute_coordination(helix,cube=50,d_max=5):
    """
    compute_coordination(helix,cube=50,d_max=5)

    Computes the coordination of each atom with each other atom.
    This function computes the bonds of the system and compte the coordination

    Parameters
    ----------
        helix : Helix
            The helix on which to compute the coordination
        cube : float, optional
            the length parameter of the cubes used to divide the system. By default 50 A
        d_max : float, optional
            the length by which each cube is enlarged to compute bonds with other cubes. By default 5 A

    Returns
    -------
        Coordination
            The list contains each type of neighbouring atom for each atom
    """

    if len(helix.Coordination): return helix.Coordination

    Types = helix.Types
    Bonds = compute_bonds_system(helix,cube=cube,d_max=d_max)

    num_at = len(Types)

    C_at = [[] for k in range(num_at)]
    C_at_check = [[] for k in range(num_at)]
    for index_1, index_2 in Bonds:
        if not (index_2 in C_at_check[index_1]):
            C_at[index_1].append(Types[index_2])
            C_at_check[index_1].append(index_2)

        if not (index_1 in C_at_check[index_2]):
            C_at[index_2].append(Types[index_1])
            C_at_check[index_2].append(index_1)

    helix.coordination = C_at

    return C_at



def compute_rdf(helix,cube=30,rdf_max=5):
    """
    compute_rdf(helix,cube=30,rdf_max=5)

    Computes the RDF distances by slicing the system into sub-cubes.

    Parameters
    ----------
        helix : Helix
        The helix on which to compute the RDF
        cube : float, optional
            the length parameter of the cubes used to divide the system. By default 30 A
        rdf_max : float, optional
            the length by which each cube is enlarged to compute distances with other cubes. By default 5 A

    Returns
    -------
        RDF
            The RDF as a list of list. Each list contains the RDF of the type compared with the other types
    """
    N_th = shanc.nprocs

    Pos = helix.Pos
    Types = helix.Types
    Lims = helix.Lims
    periodic = helix.periodic

    Lx, Ly, Lz = np.max(Pos, axis=0)
    lx, ly, lz = np.min(Pos, axis=0)

    # Use the system bounds from Pos unless periodic limits are explicitly provided.
    if periodic:
        if len(Lims) == 0:
            print("No limits provided – running as non-periodic.")
            periodic = False
        else:
            lz, Lz = Lims[2]

    # Divide the cell into cubic subregions for scalable neighbor counting.
    Nx = int((Lx - lx) // cube + 1)
    Ny = int((Ly - ly) // cube + 1)
    Nz = int((Lz - lz) // cube + 1)

    Pos_added   = np.copy(Pos)
    Types_added = np.copy(Types)


    if periodic:
        Pos_add_z    = Pos[:, 2] > (Lz - rdf_max)
        Pos_remove_z = Pos[:, 2] < (lz + rdf_max)

        Pos_added = np.vstack([
            Pos_added,
            Pos[Pos_add_z]    - [0, 0, Lz - lz],
            Pos[Pos_remove_z] + [0, 0, Lz - lz],
        ])
        Types_added = np.concatenate([
            Types_added,
            Types[Pos_add_z],
            Types[Pos_remove_z],
        ])

    max_types = int(np.max(Types))


    x_pos = Pos_added[:,0]
    y_pos = Pos_added[:,1]
    z_pos = Pos_added[:,2]

    Rdf = [ [[] for k in range(max_types)] for i in range(max_types) ]
    def evaluate_rdf(Rdf,Nz_range):
        """function used to evaluate the rdf with multithreading support"""

        for z in range(Nz_range[0],Nz_range[1]):
            z_select_pos_p = (Pos_added[:, 2] >= z*cube + lz - rdf_max) & (Pos_added[:, 2] < (z+1)*cube + lz + rdf_max)
            z_select_pos_u = (Pos_added[:, 2] >= z*cube + lz) & (Pos_added[:, 2] < (z+1)*cube + lz)

            for y in range(Ny):
                y_select_pos_p = (y_pos >= y*cube + ly - rdf_max) & (y_pos < (y+1)*cube + ly + rdf_max)
                y_select_pos_u = (y_pos >= y*cube + ly) & (y_pos < (y+1)*cube + ly)

                for x in range(Nx):
                    #Extract small cubes to compute the bonds
                    mask_p = z_select_pos_p & y_select_pos_p & (x_pos >= x*cube + lx - rdf_max) & (x_pos < (x+1)*cube + lx + rdf_max)
                    mask_u = z_select_pos_u & y_select_pos_u & (x_pos >= x*cube + lx) & (x_pos < (x+1)*cube + lx)

                    #Two truncation are made
                    # One for the system + rdf distance to make sure all atoms are taken into account
                    # the other one is just the atoms where the rdf is computed
                    Pos_trunc_large   = Pos_added[mask_p]
                    Pos_trunc_small   = Pos_added[mask_u]
                    Types_trunc_large   = Types_added[mask_p]
                    Types_trunc_small   = Types_added[mask_u]

                    Distances = sd.cdist(Pos_trunc_small,Pos_trunc_large)
                    for type_1 in range(max_types):
                        for type_2 in range(max_types):
                            T1 = (Types_trunc_small == (type_1+1))
                            T2 = (Types_trunc_large == (type_2+1))


                            D_rdf = Distances[T1][:,T2].flatten()
                            D_rdf = D_rdf[D_rdf <= rdf_max]
                            Rdf[type_1][type_2] = Rdf[type_1][type_2] + D_rdf.tolist()





    Nz_per_thread = [Nz // N_th] * N_th
    for j in range(Nz % N_th):
        Nz_per_thread[j] += 1

    ranges, count = [], 0
    for n in Nz_per_thread:
        ranges.append([count, count + n])
        count += n

    threads = []
    for rng in ranges:
        if rng[0] != rng[1]:
            t = th.Thread(target=evaluate_rdf, args=(Rdf,rng))
            threads.append(t)
            t.start()
    for t in threads:
        t.join()

    helix.rdf = Rdf
    return Rdf



def _convert_symbol_to_type(type_map,symbol):
    """
    convert a symbol to multiple types if possible
    """
    if type_map == {}: raise ValueError("A symbol was used to select atom types, but no mapping was provided")
    else:
        T = []
        for ty in type_map:
            if type_map[ty] == symbol:
                T.append(ty)
    return T

def _clean_type(helix,ty):
    """
    clean the types to be more usable
    """
    T = []
    label = ""
    if type(ty) == int:
        T = [ty]
        label = str(ty)
    elif type(ty) == list:
        for t in ty:
            if type(t) == str:
                t = _convert_symbol_to_type(helix.type_map,t)
            T = T+t
            label += str(t)+","
        label = label[:-1]

    elif type(ty) == str:
        ty = ty.split(",")
        for t in ty:
            t = _convert_symbol_to_type(helix.type_map,t)
            T = T+t
        label = ty

    return T,label


def plot_rdf(self,type_1,type_2,cube=50, rdf_max=5, density=True, vline=None, title=None, plot=False, save=True, file_name="rdf.svg"):
    """
    plot_rdf(type_1,type_2,cube=50, rdf_max=5, density=True, vline=None, title=None, plot=False, save=True, file_name="rdf.svg")

    Plot the RDF between type_1 and type_2. The types can be either number, list of numbers, atomic symbols or list of atomic symbols.
    Examples : [0,1] selects the type 1 and 2. "O" selects all the oxygens. "O,H" selects the oxygens and hydrogens.

    Parameters
    ----------
        type_1 : int or str or list
            the selection of the atoms on which to do the RDF. See above for examples.
        type_2 : int or str or list
            the selection of the atoms that are detected for the RDF. See above for examples.
        cube : float, optional
            the length parameter of the cubes used to divide the system. By default 50 A
        rdf_max : float, optional
            the length by which each cube is enlarged to compute bonds with other cubes. By default 5 A
        density : bool, optional
            If one wants to compute RDF as a density or as a number. True by default
        vline : float, optional
            Put a vertical line at that point. None by default
        title : str, optional
            If variable is set, a title for the graph is generated. None by default
        plot : bool, optional
            Show the graph. False by default
        save : bool, optional
            save the graph or not. True by default
        file_name : str, optional
            the file name if the file is saved. "rdf.svg" by default

    Returns
    -------
        None
            The graph is either generated or showed or both.
    """


    rdf = self.rdf
    if not len(rdf):
        rdf = compute_rdf(self,cube=cube,rdf_max=rdf_max)

    T1,label_1 = _clean_type(self,type_1)
    T2,label_2 = _clean_type(self,type_2)


    Dist_list = []
    count = 0
    for t1 in T1:
        for t2 in T2:
            Dist_list += rdf[t1-1][t2-1]
    for t2 in T2:
        count += np.sum(self.Types==t2)


    purple           = _PURPLE
    dark_purple      = _DARK_PURPLE
    dark_dark_purple = _DARK_DARK_PURPLE

    fig,ax = plt.subplots(figsize=(8, 6), dpi=200)
    plt.rcParams.update({'font.size': 15})
    plt.rcParams['svg.fonttype'] = 'none'

    counts, edges, patches = plt.hist(
        Dist_list, bins=100, range=(0, rdf_max),
        color=purple, edgecolor=dark_purple, linewidth=1,
    )

    if density:
        radius = ((np.roll(edges, 1) + edges) / 2)[1:]
        dr  = radius[1] - radius[0]
        g_r = counts / ((4 * np.pi * radius**2 * dr) * count)
        for rect, val in zip(patches, g_r):
            rect.set_height(val)
        plt.ylabel("g(r)", color=purple,size=17)
        ax.set_ylim(0, np.max(g_r)*1.2)
    else:
        plt.ylabel("Number", color=purple,size=17)

    if title:
        plt.title(title, color=purple)

    plt.xlabel("Distance (Å)", color=purple,size=17)

    if vline:
        tick_locs = [k for k in range(int(rdf_max))] + [vline]
        ax.axvline(vline, color=dark_dark_purple, linestyle='--', linewidth=3,
               label=f"Bond distance {vline} Å")
    else:
        tick_locs = [k for k in range(int(rdf_max))]

    plt.xticks(tick_locs,color=purple,size=17)
    plt.yticks(color=purple,size=17)
    plt.xlim(0, rdf_max)

    plt.tight_layout()
    if save:
        plt.savefig(file_name)
    if plot:
        plt.show()

Helix.plot_rdf = plot_rdf


def plot_coordination(self, type_at, cube=30, d_max=5, decompose=False, type_colors={}, title=None, plot=False, save=True, file_name="coordination.svg"):
    """
    plot_coordination(type_at, cube=30, d_max=5, decompose=False, type_colors={}, title=None, plot=False, save=True, file_name="coordination.svg")

    Plot the coordination for a selection of atom. The type_at is the variable to set for the selection.
    Examples : [0,1] selects the type 1 and 2. "O" selects all the oxygens. "O,H" selects the oxygens and hydrogens.

    Parameters
    ----------
        type_at : int or str or list
            the selection of the atoms. See above for examples.
        cube : float, optional
            the length parameter of the cubes used to divide the system. By default 50 A
        d_max : float, optional
            the length by which each cube is enlarged to compute bonds with other cubes. By default 5 A
        decompose : bool, optional
            whether to decompose each bar per atom type. False by default
        type_colors : dict, optional
            Specify the color of each type for the decomposition. {} by default
        title : str, optional
            If variable is set, a title for the graph is generated. None by default
        plot : bool, optional
            Show the graph. False by default
        save : bool, optional
            save the graph or not. True by default
        file_name : str, optional
            the file name if the file is saved. "coordination.svg" by default

    Returns
    -------
        None
            The graph is either generated or showed or both.
    """

    T,label = _clean_type(self,type_at)
    Types = self.Types
    mask = np.zeros((len(Types))).astype("bool")
    for ty in T:
        mask += (Types==ty)

    do_colors = len(type_colors)==0
    type_map = self.type_map


    if type_map != {}:
        for type in type_map:
            symbol = type_map[type]
            colour = ELEMENT_COLOURS[symbol]
            if do_colors:
                type_colors[type] = colour
    else:
        #Hopefully there are no more than 4 types of atom........
        type_colors = {1: 'gold', 2: 'red', 3: 'blue', 4: [255, 255, 0]}

    Coordination = compute_coordination(self,cube=cube,d_max=d_max)


    max_len = 0
    min_len = 100
    for at_list,is_in in zip(Coordination,mask):
        if is_in and len(at_list) > max_len:
            max_len = len(at_list)
        if is_in and len(at_list) < min_len:
            min_len = len(at_list)

    max_type = int(np.max(Types))

    Hist = np.zeros((max_len+1))
    Hist_at = np.zeros((max_len+1,max_type))

    for at,at_list in zip(range(len(Coordination)),Coordination):
        if mask[at]:
            count = len(at_list)
            Hist[count] += 1
            for att in at_list:
                Hist_at[count,att-1] += 1



    purple           = _PURPLE
    dark_purple      = _DARK_PURPLE
    dark_dark_purple = _DARK_DARK_PURPLE

    fig,ax = plt.subplots(figsize=(8, 6), dpi=200)
    plt.rcParams.update({'font.size': 15})
    plt.rcParams['svg.fonttype'] = 'none'

    if title:
        plt.title(title, color=purple)

    # plt.hist(Hist,bins=2*max_len,range=(0,max_len),color=purple,edge_color=dark_purple,linewidth=1)
    if decompose:
        counts = np.zeros((max_type,max_len+1))
        Hist_at = Hist_at.transpose()
        counts[-1] = Hist_at[-1]

        for index_hist, hist_ in zip(range(1,max_type),Hist_at[::-1][1:]):
            index_ = max_type - index_hist -1
            counts[index_] = counts[index_ + 1] + hist_

        for index, counts_ in zip(range(max_len),counts):
            color = np.array(type_colors[index+1])/255
            color_edge = [max(color[0]-100/255,0),max(color[1]-100/255,0),max(color[2]-100/255,0)]
            plt.bar(np.arange(max_len+1),counts_,color=color,edgecolor=color_edge,label="Type {}".format(index+1))
        plt.legend()

    else:
        plt.bar(np.arange(max_len+1),Hist,color=dark_purple)


    plt.xlabel("Number of bonds", color=purple, size=17)
    plt.ylabel("Number of atoms", color=purple, size=17)


    plt.xticks(np.arange(min_len,max_len+1),color=purple,size=17)
    plt.yticks(color=purple,size=17)
    plt.xlim(min_len-0.5, max_len+0.5)

    plt.tight_layout()
    if save:
        plt.savefig(file_name)
    if plot:
        plt.show()

Helix.plot_coordination = plot_coordination







# ---------------------------------------------------------------------------
# Defect analysis / saving
# ---------------------------------------------------------------------------

# def save_defects(self, file_name="defects.txt",
#                  cation_type=1, anion_type=2):
#     """
#     Re-map atom types to encode coordination defects and write an XYZ file.

#     Normal coordination (type 1 → 1, type 2 → 2) is preserved; under- and
#     over-coordinated atoms receive higher type ids for easy visualisation.
#     """

#     print("TO REview")

#     Pos = self.Pos
#     Types = self.Types
#     Lims = self.Lims
#     periodic = self.periodic

#     _, Cat_count_An, An_count_Cat = compute_rdf(
#         Pos, Types, cube=30, periodic=periodic, Lims=Lims, rdf_max=5,
#         cation_type=cation_type, anion_type=anion_type,
#     )

#     # Cation re-mapping: coord 4 → type 1 (normal); others → types 5-12
#     for coord, new_type in [(4, 1), (0, 5), (1, 6), (2, 7), (3, 8),
#                              (5, 9), (6, 10), (7, 11), (8, 12)]:
#         Types[Types == cation_type] = np.where(
#             Cat_count_An == coord, new_type, Types[Types == cation_type]
#         )

#     # Anion re-mapping: coord 2 → type 2 (normal); others → types 13-18
#     for coord, new_type in [(2, 2), (0, 13), (1, 14), (3, 15),
#                              (4, 16), (5, 17), (6, 18)]:
#         Types[Types == anion_type] = np.where(
#             An_count_Cat == coord, new_type, Types[Types == anion_type]
#         )

#     write_xyz(file_name, Pos, Types)

# Helix.save_defects = save_defects



# ---------------------------------------------------------------------------
# Visualisation – atom rendering
# ---------------------------------------------------------------------------

def validate_unit_cell(Atom_pos, Atom_types, Lims,
                       expected_dist=1.613, tol=0.1,
                       type1=1, o_type=2, o_neighbours=4):
    """
    Verify that every type-1 atom has exactly o_neighbours type-2 neighbours
    at expected_dist ± tol (Å), using minimum-image periodic boundary conditions.
                       type1=1, o_type=2, o_neighbours=4):
    """
    L = np.array([Lims[0][1] - Lims[0][0],
                  Lims[1][1] - Lims[1][0],
                  Lims[2][1] - Lims[2][0]])

    Pos_type1 = Atom_pos[Atom_types == type1]
    Pos_o     = Atom_pos[Atom_types == o_type]

    all_ok = True
    for i, t1 in enumerate(Pos_type1):
        bonds = []
        for o in Pos_o:
            delta = o - t1
            delta -= L * np.round(delta / L)
            d = np.linalg.norm(delta)
            if abs(d - expected_dist) < tol:
                bonds.append(round(d, 4))
        if len(bonds) != o_neighbours:
            print(f"  [!] type-{type1} atom {i}: {len(bonds)} bond(s) at {bonds} "
                  f"— expected {o_neighbours} × {expected_dist} Å")
            all_ok = False

    if all_ok:
        print(f"  Unit cell OK: all type-{type1} have {o_neighbours} type-{o_type} "
              f"neighbours around {expected_dist} Å")
    return all_ok

def _plot_atoms(helix,plotter,type_colors={},type_radius={},sphere=True,do_fancy=True,do_cast=False):
    """
    """
    Types = helix.Types
    Pos = helix.Pos

    Pos_cast = helix.Pos_cast
    Types_cast = helix.Types_cast

    is_cast = len(Pos_cast) > 0


    do_colors = len(type_colors)==0
    do_radius = len(type_radius)==0

    type_map = helix.type_map

    if type_map != {}:
        for type in type_map:
            symbol = type_map[type]
            colour = ELEMENT_COLOURS[symbol]
            if do_colors:
                type_colors[type] = colour
            radius = ELEMENT_RADIUS[symbol]
            if do_radius:
                type_radius[type] = radius/2
    else:
        #Hopefully there are no more than 4 types of atom........
        type_colors = {1: 'gold', 2: 'red', 3: 'blue', 4: [255, 255, 0]}
        type_radius = {1: 0.5, 2: 0.4, 3: 0.8, 4: 0.8}


    for type in type_colors:
        radius = type_radius[type]
        color = type_colors[type]
        mask = (Types == type)
        if not np.any(mask):
            continue
        if sphere:
            if do_fancy:
                geom = pv.Sphere(radius=radius,theta_resolution=8,phi_resolution=8)
            else: geom = pv.Sphere(radius=radius,theta_resolution=5,phi_resolution=5)
            mesh = pv.PolyData(Pos[mask].astype(float)).glyph(scale=False, geom=geom, orient=False)
        else:
            mesh = pv.PolyData(Pos[mask].astype(float))

        if do_fancy: plotter.add_mesh(mesh, color=color, pbr=True,roughness=.5,metallic=.2)
        else: plotter.add_mesh(mesh, color=color, lighting=False)

        if is_cast and do_cast:
            mask = (Types_cast == type)
            if not np.any(mask): continue
            color_cast = [max(color[0]-125,0),max(color[1]-125,0),max(color[2]-125,0)]
            if sphere:
                geom = pv.Sphere(radius=radius)
                mesh = pv.PolyData(Pos_cast[mask].astype(float)).glyph(scale=False, geom=geom, orient=False)
            else:
                mesh = pv.PolyData(Pos_cast[mask].astype(float))
            if do_fancy: plotter.add_mesh(mesh, color=color_cast, pbr=True,roughness=.5,metallic=.2)
            else: plotter.add_mesh(mesh, color=color_cast, lighting=False)




def plot_structure(self, type_colors={}, type_radius={},
                        sphere=True, parallel_proj=True,
                        draw_bonds=False, periodic_bonds=False,
                        do_fancy=False,do_cast=False):
    """Render the atomic structure in 3D using PyVista.

    Parameters
    ----------
    Pos : ndarray
        Atom positions.
    Types : ndarray
        Atom type ids.
    Lims : list, optional
        Box limits for drawing the simulation cell.
    point_size : int, optional
        Size of rendered atoms.
    type_colors : dict, optional
        Mapping from type id to display color.
    sphere : bool, optional
        Render atoms as spheres when True.
    parallel_proj : bool, optional
        Use parallel projection.
    draw_bonds : bool, optional
        Draw bonds between unlike atom types.
    max_bond_dist : float, optional
        Distance cutoff used for bond drawing.
    """
    Pos = self.Pos
    Types = self.Types
    Lims = self.Lims
    periodic=self.periodic


    Pos_cast = self.Pos_cast
    Types_cast = self.Types_cast

    plotter = pv.Plotter()
    _plot_atoms(self,plotter,type_colors=type_colors,type_radius=type_radius,sphere=sphere,do_fancy=do_fancy,do_cast=do_cast)

    if draw_bonds:

        mesh = pv.PolyData(Pos.astype(float))
        periodic = self.periodic
        if not periodic_bonds:
            self.periodic=False
        Bonds = compute_bonds_system(self)
        if not periodic_bonds:
            self.periodic=periodic

        if len(Bonds):
            lines = np.hstack([[2, i_1, i_2] for i_1, i_2 in Bonds])
            mesh.lines = lines

            if do_fancy:
                tube = mesh.tube(radius=0.1,n_sides=5)
                plotter.add_mesh(tube, color="gray", pbr=True,roughness=.5,metallic=.2,line_width=5)
            else:
                plotter.add_mesh(mesh, color="gray", lighting=False,line_width=5)

        if periodic_bonds != periodic:
            self.Bonds_analysis = []


    if Lims is not None:
        plotter.add_mesh(
            pv.Box(bounds=(Lims[0][0], Lims[0][1],
                           Lims[1][0], Lims[1][1],
                           Lims[2][0], Lims[2][1])),
            style="wireframe", color="black",
        )

    if parallel_proj:
        plotter.enable_parallel_projection()
    plotter.show()

Helix.plot_structure = plot_structure




def plot_structure_coordination(self, type_colors={}, type_radius={},
                        sphere=True,
                        draw_bonds=False, cube=50,d_max=5, periodic_bonds=False,
                        do_fancy=False):
    """Render the atomic structure in 3D using PyVista.

    Parameters
    ----------
    Pos : ndarray
        Atom positions.
    Types : ndarray
        Atom type ids.
    Lims : list, optional
        Box limits for drawing the simulation cell.
    point_size : int, optional
        Size of rendered atoms.
    type_colors : dict, optional
        Mapping from type id to display color.
    sphere : bool, optional
        Render atoms as spheres when True.
    parallel_proj : bool, optional
        Use parallel projection.
    draw_bonds : bool, optional
        Draw bonds between unlike atom types.
    max_bond_dist : float, optional
        Distance cutoff used for bond drawing.
    """

    print("THIS FUNCTION IS NOT FINISHED YET")
    Pos = self.Pos
    Types = self.Types
    Lims = self.Lims
    periodic=self.periodic


    Pos_cast = self.Pos_cast
    Types_cast = self.Types_cast
    is_cast = len(Pos_cast) > 0


    ##TO ADD CAST PLOTTING
    #
    do_colors = len(type_colors)==0
    do_radius = len(type_radius)==0



    type_map = self.type_map

    if type_map != {}:
        for type in type_map:
            symbol = type_map[type]
            colour = ELEMENT_COLOURS[symbol]
            if do_colors:
                type_colors[type] = colour
            radius = ELEMENT_RADIUS[symbol]
            if do_radius:
                type_radius[type] = radius/2

    else:
        #Hopefully there are no more than 4 types of atom........
        type_colors = {1: 'gold', 2: 'red', 3: 'blue', 4: [255, 255, 0]}
        type_radius = {1: 0.5, 2: 0.4, 3: 0.8, 4: 0.8}

    plotter = pv.Plotter()

    Coordination = compute_coordination(self,cube=cube,d_max=d_max)

    Coordination_len = np.zeros((len(Coordination)),dtype="int")


    for at_list,i_at in zip(Coordination,range(len(Coordination))):
        Coordination_len[i_at] = len(at_list)
    max_len = np.max(Coordination_len)




    for type in type_colors:
        radius = type_radius[type]
        color = type_colors[type]
        mask = (Types == type)
        if not np.any(mask):
            continue
        if do_fancy:
            geom = pv.Sphere(radius=radius,theta_resolution=8,phi_resolution=8)
        else: geom = pv.Sphere(radius=radius,theta_resolution=5,phi_resolution=5)


        for num_at in range(max_len):
            if num_at > 6:
                mask_num_at = (Coordination_len == num_at) * mask

                if sphere:
                    mesh = pv.PolyData(Pos[mask_num_at].astype(float)).glyph(scale=False, geom=geom, orient=False)
                else:
                    mesh = pv.PolyData(Pos[mask_num_at].astype(float))

                if do_fancy: plotter.add_mesh(mesh, color=color, pbr=True,roughness=.5,metallic=.2,opacity=(num_at/max_len)**3)
                else: plotter.add_mesh(mesh, color=color, lighting=False,opacity=num_at/max_len)

                if is_cast:
                    mask = (Types_cast == type)
                    if not np.any(mask): continue
                    color_cast = [max(color[0]-125,0),max(color[1]-125,0),max(color[2]-125,0)]
                    if sphere:
                        geom = pv.Sphere(radius=radius)
                        mesh = pv.PolyData(Pos_cast[mask].astype(float)).glyph(scale=False, geom=geom, orient=False)
                    else:
                        mesh = pv.PolyData(Pos_cast[mask].astype(float))
                    if do_fancy: plotter.add_mesh(mesh, color=color_cast, pbr=True,roughness=.5,metallic=.2)
                    else: plotter.add_mesh(mesh, color=color_cast, lighting=False)



    if draw_bonds:

        mesh = pv.PolyData(Pos.astype(float))
        periodic = self.periodic
        if not periodic_bonds:
            self.periodic=False
        Bonds = compute_bonds_system(self)
        if not periodic_bonds:
            self.periodic=periodic

        lines = np.hstack([[2, i_1, i_2] for i_1, i_2 in Bonds])
        mesh.lines = lines



        if do_fancy:
            tube = mesh.tube(radius=0.1,n_sides=5)
            plotter.add_mesh(tube, color="gray", pbr=True,roughness=.5,metallic=.2,line_width=5)
        else:
            plotter.add_mesh(mesh, color="gray", lighting=False,line_width=5)



    if Lims is not None:
        plotter.add_mesh(
            pv.Box(bounds=(Lims[0][0], Lims[0][1],
                           Lims[1][0], Lims[1][1],
                           Lims[2][0], Lims[2][1])),
            style="wireframe", color="black",
        )

    plotter.show()

Helix.plot_structure_coordination = plot_structure_coordination

#####
# Surface analysis
# ########


class _slider_surface:
    def __init__(self,func,d_spacing,alpha,isovalue,smoothing,lims,opacity):
        self.func = func
        self.d_spacing = float(d_spacing)
        self.alpha = float(alpha)
        self.isovalue = float(isovalue)
        self.smoothing = float(smoothing)
        self.lims = float(lims)
        self.opacity = opacity
        self.cube = None

    def __call__(self,called,value):
        if called=="d_spacing":
            self.d_spacing = value
            self.cube = self.func(self.d_spacing,self.alpha,self.isovalue,self.smoothing,self.lims,self.opacity)
        elif called=="alpha":
            self.alpha = value
            self.cube = self.func(self.d_spacing,self.alpha,self.isovalue,self.smoothing,self.lims,self.opacity)
        elif called=="isovalue":
            self.isovalue = value
            self.cube = self.func(self.d_spacing,self.alpha,self.isovalue,self.smoothing,self.lims,self.opacity,cube=self.cube)
        elif called == "smoothing":
            self.smoothing = value
            self.cube = self.func(self.d_spacing,self.alpha,self.isovalue,self.smoothing,self.lims,self.opacity,cube=self.cube)
        elif called == "lims":
            self.lims = value
            self.cube = self.func(self.d_spacing,self.alpha,self.isovalue,self.smoothing,self.lims,self.opacity,cube=self.cube)
        elif called == "opacity":
            self.opacity = value
            self.cube = self.func(self.d_spacing,self.alpha,self.isovalue,self.smoothing,self.lims,self.opacity,cube=self.cube)




def _compute_quick_surface(Pos, grid, Lims, alpha=2, prec=5, d=5,
                           length_box=50):
    """
    Compute a Gaussian-density isosurface (VMD QuickSurf-style).

    This function is used to compute surface like in the quicksurf of VMD
    The surface is computed using sum of gaussians centered on each atom
    \\rho = \\sum e^{-\\frac{\\abs{r-r_i}^2}{2\\alpha^2}}
    For more information see : https://diglib.eg.org/items/d34c2b92-60b4-489b-8801-9b8cad2d8ba2

    To accelerate the computation, the system is divided in multiple boxes in which the distance between
    the atoms and the grid is computed. Each box is larger for the atoms than for the grid in order not
    to have border effects.

    It also supports multithreading

    Parameters
    ----------
    Pos        : (N, 3) array – atom positions
    grid       : pv.ImageData – grid on which to evaluate the density
    Lims       : [[Lx,lx],[Ly,ly],[Lz,lz]] – system extents
    alpha      : float – Gaussian width (Å), default 2
    prec       : float – atom-box padding beyond grid-box edge (Å), default 20
    d          : float – system extent expansion (Å), default 10
    length_box : float – box edge length (Å), default 20
    N_th       : int   – number of threads, default 8

    Returns
    -------
    cube : (M,) array – density values at each grid point
    """
    N_th = shanc.nprocs

    x, y, z  = grid.points.T
    Pos_Grid = grid.points
    # Evaluate in parallel over z-slab ranges to accelerate the surface
    # density computation on large grids.

    Lx, lx = Lims[0]
    Ly, ly = Lims[1]
    Lz, lz = Lims[2]

    Nx_box = int((Lx - lx + 2*d) / length_box) + 1
    Ny_box = int((Ly - ly + 2*d) / length_box) + 1
    Nz_box = int((Lz - lz + 2*d) / length_box) + 1




    x_pos, y_pos, z_pos = Pos[:,0], Pos[:,1], Pos[:,2]
    # import time
    def evaluate_surface(cube, Nx_box, Ny_box, Nz_range):
        # AA, BB,CC,DD,EE =0,0,0,0,0
        # aaaa = time.time()
        for z_box in range(Nz_range[0], Nz_range[1]):
            z_select_box = (z >= z_box*length_box + lz - d) & (z <= (z_box+1)*length_box + lz + d)
            z_select_pos = (z_pos >= z_box*length_box - prec + lz - d) & (z_pos <= (z_box+1)*length_box + prec + lz + d)
            for y_box in range(Ny_box):
                y_select_box = (y >= y_box*length_box + ly - d) & (y <= (y_box+1)*length_box + ly + d)
                y_select_pos = (y_pos >= y_box*length_box - prec + ly - d) & (y_pos <= (y_box+1)*length_box + prec + ly + d)
                for x_box in range(Nx_box):
                    # aa = time.time()
                    Ind_Box = (x >= x_box*length_box + lx - d) & (x <= (x_box+1)*length_box + lx + d) & y_select_box & z_select_box
                    Pos_Box = Pos_Grid[Ind_Box]
                    # bb=time.time()
                    Ind_trunc = (x_pos >= x_box*length_box - prec + lx - d) & (x_pos <= (x_box+1)*length_box + prec + lx + d)  & y_select_pos & z_select_pos
                    Pos_trunc = Pos[Ind_trunc]
                    # cc = time.time()
                    Dist = sd.cdist(Pos_Box, Pos_trunc)
                    # dd = time.time()
                    # ee=  time.time()
                    cube[Ind_Box] = np.einsum("ij->i", np.exp(-Dist**2 / (2 * alpha**2)))
                    # ff = time.time()
                    # AA+= bb-aa
                    # BB+= cc-bb
                    # CC+=dd-cc
                    # DD+=ee-dd
                    # EE+=ff-ee

        # print(AA,BB,CC,DD,EE)
        # print(time.time()-aaaa)


    cube = np.zeros(len(Pos_Grid))

    Nz_per_thread = [Nz_box // N_th] * N_th
    for j in range(Nz_box % N_th):
        Nz_per_thread[j] += 1

    ranges, count = [], 0
    for n in Nz_per_thread:
        ranges.append([count, count + n])
        count += n

    threads = []
    for rng in ranges:
        if rng[0] != rng[1]:
            t = th.Thread(target=evaluate_surface, args=(cube, Nx_box, Ny_box, rng))
            threads.append(t)
            t.start()
    for t in threads:
        t.join()

    return cube


def _compute_density(Pos_points_surface, Pos, periodic=False, Lims = [], length_box=20,d_max = 5):
    """
    Function to compute the density from the points on the surface
    """
    N_th = shanc.nprocs

    Lx, Ly, Lz = np.max(Pos_points_surface, axis=0)
    lx, ly, lz = np.min(Pos_points_surface, axis=0)


    # Divide the cell into cubic subregions for scalable neighbor counting.
    Nx_box = int((Lx - lx) // length_box + 1)
    Ny_box = int((Ly - ly) // length_box + 1)
    Nz_box = int((Lz - lz) // length_box + 1)



    Density = np.zeros(len(Pos_points_surface))

    pos_x = Pos[:,0]
    pos_y = Pos[:,1]
    pos_z = Pos[:,2]

    pos_surf_x = Pos_points_surface[:,0]
    pos_surf_y = Pos_points_surface[:,1]
    pos_surf_z = Pos_points_surface[:,2]



    def evaluate_surface(Density, Nx_box, Ny_box, Nz_range):
        for z in range(Nz_range[0],Nz_range[1]):
            select_z = (pos_z >= z*length_box + lz - d_max) & (pos_z < (z+1)*length_box + lz + d_max)
            select_z_surf = (pos_surf_z >= z*length_box + lz) & (pos_surf_z < (z+1)*length_box + lz)
            for y in range(Ny_box):
                select_y = (pos_y >= y*length_box + ly - d_max) & (pos_y < (y+1)*length_box + ly + d_max)
                select_y_surf = (pos_surf_y >= y*length_box + ly) & (pos_surf_y < (y+1)*length_box + ly)
                for x in range(Nx_box):
                    mask_pos = (pos_x >= x*length_box + lx - d_max) & (pos_x < (x+1)*length_box + lx + d_max) & select_y & select_z
                    mask_surf = (pos_surf_x >= x*length_box + lx) & (pos_surf_x < (x+1)*length_box + lx) & select_y_surf & select_z_surf

                    Pos_large = Pos[mask_pos]
                    Pos_surf_trunc = Pos_points_surface[mask_surf]
                    Dist = sd.cdist(Pos_surf_trunc,Pos_large)

                    Dist = Dist<d_max

                    Density[mask_surf] = np.sum(Dist,axis=1)



    Nz_per_thread = [Nz_box // N_th] * N_th
    for j in range(Nz_box % N_th):
        Nz_per_thread[j] += 1

    ranges, count = [], 0
    for n in Nz_per_thread:
        ranges.append([count, count + n])
        count += n

    threads = []
    for rng in ranges:
        if rng[0] != rng[1]:
            t = th.Thread(target=evaluate_surface, args=(Density, Nx_box, Ny_box, rng))
            threads.append(t)
            t.start()
    for t in threads:
        t.join()


    if np.max(Density) == 0:
        return Density

    Density  = Density / np.max(Density)

    return Density


def plot_density(self,d=5,prec=10,length_box=20,d_spacing_init=3,isovalue_init=1,alpha_init=2.,smoothing_init=3000,d_max_init=5,plot_atoms=True):

    """
    analyze_defects(Pos,Types,periodic=False,Lims=[],Cycles=None,L_cycles=None)


    """

    #########MIGHT BE POSSIBLE TO USE RECOMPUTED PARAMETERS TO GET THIS HELIX
    #
    N_th = shanc.nprocs

    Pos = self.Pos
    Lims = self.Lims
    periodic = self.periodic


    plotter = pv.Plotter()
    plotter.add_axes()


    Lx,Ly,Lz = np.max(Pos,axis=0) + d
    lx,ly,lz = np.min(Pos,axis=0) - d


    def slider_precision(d_spacing,alpha,isovalue,smoothing,lims,opacity,cube=None):

        Nx = int(round((Lx-lx+2*d)/d_spacing))+1
        Ny = int(round((Ly-ly+2*d)/d_spacing))+1
        Nz = int(round((Lz-lz+2*d)/d_spacing))+1

        grid = pv.ImageData(dimensions=(Nx,Ny,Nz),origin=(lx-d,ly-d,lz-d),spacing=(d_spacing,d_spacing,d_spacing))
        Lims_grid = [[Lx,lx],[Ly,ly],[Lz,lz]]

        if type(cube) is type(None):
            #No cube were provided, therefore compute the whole cube again
            #This is used to not recompute the cube everytime you change the isovalue
            cube = _compute_quick_surface(Pos,grid,Lims_grid,alpha=alpha,prec=prec,d=d,length_box=length_box)
        #this is to make sure that it is a float otherwise problem arises
        isovalue = float(isovalue)
        contour = grid.contour(isosurfaces=(isovalue),scalars=cube)

        if smoothing != 0:
            smooth = contour.smooth(n_iter=int(smoothing))
        else: smooth = contour

        Pos_points_surface = smooth.points
        Density = _compute_density(Pos_points_surface,Pos,periodic=periodic,Lims=Lims, length_box=length_box,d_max=lims)


        if plot_atoms:
            plotter.add_mesh(smooth,name="contour",opacity=opacity,cmap="cool",scalars=Density,scalar_bar_args={"title":"Density"})

        else:
            plotter.add_mesh(smooth,name="contour",opacity=1.0,cmap="cool",scalars=Density,scalar_bar_args={"title":"Density"})




        return cube



    slider = _slider_surface(slider_precision,10,3,1,0,1,1)

    plotter.add_slider_widget(lambda value: slider("alpha",value), [0.5, 10],value=alpha_init,title="Alpha", fmt="%1.1f",pointa=(0.01,.8),pointb=(0.2,.8),slider_width=0.02,title_height=0.02)
    plotter.add_slider_widget(lambda value: slider("isovalue",value), [0.5, 20],value=isovalue_init,title="Isovalue", fmt="%1.1f",pointa=(0.01,.7),pointb=(0.2,.7),slider_width=0.02,title_height=0.02)
    plotter.add_slider_widget(lambda value: slider("smoothing",value), [0, 5000],value=smoothing_init,title="Smoothing", fmt="%1.1f",pointa=(0.01,.6),pointb=(0.2,.6),slider_width=0.02,title_height=0.02)
    plotter.add_slider_widget(lambda value: slider("lims",value), [0.5, 10],value=d_max_init,title="Dist max density", fmt="%1.1f",pointa=(0.01,.5),pointb=(0.2,.5),slider_width=0.02,title_height=0.02)
    if plot_atoms:
        plotter.add_slider_widget(lambda value: slider("opacity",value), [0, 1],value=0.7,title="Opacity", fmt="%1.1f",pointa=(0.01,.4),pointb=(0.2,.4),slider_width=0.02,title_height=0.02)
        _plot_atoms(self,plotter)
    plotter.add_slider_widget(lambda value: slider("d_spacing",value), [0.5, 10],value=d_spacing_init,title="Spacing", fmt="%1.1f",pointa=(0.01,.9),pointb=(0.2,.9),slider_width=0.02,title_height=0.02)

    plotter.show()

Helix.plot_density = plot_density


def _blur_colors(Pos_points_surface, colors, periodic=False, Lims = [], length_box=20,d_max = 5):
    """
    Function to blur the colors on the surface
    """
    #Do a Blur
    colors_blur = np.zeros(np.shape(colors))
    #if d_max too small cubes will appear

    Lx, Ly, Lz = np.max(Pos_points_surface, axis=0)
    lx, ly, lz = np.min(Pos_points_surface, axis=0)



    # Divide the cell into cubic subregions for scalable neighbor counting.
    Nx = int((Lx - lx) // length_box + 1)
    Ny = int((Ly - ly) // length_box + 1)
    Nz = int((Lz - lz) // length_box + 1)

    Pos_added   = np.copy(Pos_points_surface)
    Colors_added = np.copy(colors)


    if periodic:
        Pos_add_z    = Pos_points_surface[:, 2] > (Lz - d_max)
        Pos_remove_z = Pos_points_surface[:, 2] < (lz + d_max)

        Pos_added = np.vstack([
            Pos_added,
            Pos_points_surface[Pos_add_z]    - [0, 0, Lz - lz],
            Pos_points_surface[Pos_remove_z] + [0, 0, Lz - lz],
        ])
        Colors_added = np.concatenate([
            colors,
            colors[Pos_add_z],
            colors[Pos_remove_z]
        ])


    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):


                mask_p = (
                    (Pos_added[:, 0] >= x*length_box + lx - d_max) & (Pos_added[:, 0] < (x+1)*length_box + lx + d_max) &
                    (Pos_added[:, 1] >= y*length_box + ly - d_max) & (Pos_added[:, 1] < (y+1)*length_box + ly + d_max) &
                    (Pos_added[:, 2] >= z*length_box + lz - d_max) & (Pos_added[:, 2] < (z+1)*length_box + lz + d_max)
                )
                mask_u = (
                    (Pos_points_surface[:, 0] >= x*length_box + lx) & (Pos_points_surface[:, 0] < (x+1)*length_box + lx) &
                    (Pos_points_surface[:, 1] >= y*length_box + ly) & (Pos_points_surface[:, 1] < (y+1)*length_box + ly) &
                    (Pos_points_surface[:, 2] >= z*length_box + lz) & (Pos_points_surface[:, 2] < (z+1)*length_box + lz)
                )

                Pos_trunc_large = Pos_added[mask_p]
                Pos_trunc_small = Pos_points_surface[mask_u]

                Dist = sd.cdist(Pos_trunc_small,Pos_trunc_large)
                Dist = (Dist<1) * 1 + (Dist>=1)*Dist

                Colors_trunc = Colors_added[mask_p]
                Colors_summed = np.einsum("ij,jk->ik",1/Dist**2,Colors_trunc) / np.expand_dims(np.einsum("ij->i",1/Dist**2),1)
                colors_blur[mask_u] = Colors_summed


    return colors_blur



def plot_curvature(self,prec=5,d=5,length_box=30,d_spacing_init=3,isovalue_init=1,alpha_init=2.,smoothing_init=200,plot_atoms=True):
    """
    Putting prec too low and/or length box too high might result in an apparition of "cubes" in the shape

    """
    Pos = self.Pos
    Lims = self.Lims
    periodic = self.periodic

    Lx,Ly,Lz = np.max(Pos,axis=0) + d
    lx,ly,lz = np.min(Pos,axis=0) - d


    def slider_precision(d_spacing,alpha,isovalue,smoothing,lims,opacity,cube=None):
        Nx = int(round((Lx-lx+2*d)/d_spacing))+1
        Ny = int(round((Ly-ly+2*d)/d_spacing))+1
        Nz = int(round((Lz-lz+2*d)/d_spacing))+1

        grid = pv.ImageData(dimensions=(Nx,Ny,Nz),origin=(lx-d,ly-d,lz-d),spacing=(d_spacing,d_spacing,d_spacing))
        Lims_grid = [[Lx,lx],[Ly,ly],[Lz,lz]]

        if type(cube) is type(None):
            #No cube were provided, therefore compute the whole cube again
            #This is used to not recompute the cube everytime you change the isovalue
            cube = _compute_quick_surface(Pos,grid,Lims_grid,alpha=alpha,prec=prec,d=d,length_box=length_box)

        #this is to make sure that it is a float otherwise problem arises
        isovalue = float(isovalue)
        contour= grid.contour(isosurfaces=(isovalue),scalars=cube)

        if smoothing != 0:
            smooth = contour.smooth(n_iter=int(smoothing))
        else: smooth = contour


        curv_min = smooth.curvature(curv_type="minimum")
        curv_max = smooth.curvature(curv_type="maximum")


        curv_max_sorted_trunc = np.sort(curv_max)[len(curv_max)//10:len(curv_max) - len(curv_max)//10]
        L = np.arange(len(curv_max_sorted_trunc))
        a,b = np.polyfit(L,curv_max_sorted_trunc,1)
        c_max_max = a * len(curv_max) + b

        curv_min_sorted_trunc = np.sort(curv_min)[len(curv_min)//10:len(curv_min) - len(curv_min)//10]
        L = np.arange(len(curv_min_sorted_trunc))
        a,b = np.polyfit(L,curv_min_sorted_trunc,1)
        c_min_max = a * len(curv_min) + b


        norm = max([c_max_max,c_min_max]) * lims

        curv_min = curv_min / norm
        curv_max = curv_max / norm


        colors = np.zeros((len(curv_min),3))

        for index,c_min,c_max in zip(range(len(curv_min)),curv_min,curv_max):
            if c_max > 1: c_max = 1
            if c_min > 1: c_min = 1
            if c_max < -1: c_max = -1
            if c_min < -1: c_min = -1

            colors[index,0] = 1 - abs((c_max*2 + abs(c_min) - c_min)/4) - (abs(c_min+c_max) + (c_min + c_max))/4
            colors[index,1] = 1 - abs((abs(c_min+c_max)-(c_min+c_max))/4) - (abs(c_min+c_max) + (c_min + c_max))/4
            colors[index,2] = 1 - abs((abs(c_min+c_max)-(c_min+c_max))/4) - abs((c_max*2 + abs(c_min) - c_min)/4)


        #Remove <0 and >1
        colors = (colors > 0) * (colors < 1) * colors + (colors>=1) * 1


        #BLUR
        Pos_points_surface = smooth.points
        colors_blur = _blur_colors(Pos_points_surface, colors, periodic=periodic, Lims = Lims, length_box= length_box,d_max = 5)

        if plot_atoms:
            plotter.add_mesh(smooth,name="contour",opacity=opacity,scalars=colors_blur,rgb=True)

        else: plotter.add_mesh(smooth,name="contour",opacity=1.0,scalars=colors_blur,rgb=True)


        return cube


    plotter = pv.Plotter()
    # plotter.add_background_image("lims.png",auto_resize=False)

    slider = _slider_surface(slider_precision,5,3,1,0,1,1)

    plotter.add_slider_widget(lambda value: slider("alpha",value), [0.5, 10],value=alpha_init,title="Alpha", fmt="%1.1f",pointa=(0.01,.8),pointb=(0.2,.8),slider_width=0.02,title_height=0.02)
    plotter.add_slider_widget(lambda value: slider("isovalue",value), [0.5, 20],value=isovalue_init,title="Isovalue", fmt="%1.1f",pointa=(0.01,.7),pointb=(0.2,.7),slider_width=0.02,title_height=0.02)
    plotter.add_slider_widget(lambda value: slider("smoothing",value), [0, 1000],value=smoothing_init,title="Smoothing", fmt="%1.1f",pointa=(0.01,.6),pointb=(0.2,.6),slider_width=0.02,title_height=0.02)
    plotter.add_slider_widget(lambda value: slider("lims",value), [0.5, 2],value=1.,title="Lims", fmt="%1.1f",pointa=(0.01,.5),pointb=(0.2,.5),slider_width=0.02,title_height=0.02)
    if plot_atoms:
        _plot_atoms(self,plotter)
        plotter.add_slider_widget(lambda value: slider("opacity",value), [0,1],value=0.9,title="Opacity", fmt="%1.1f",pointa=(0.01,.4),pointb=(0.2,.4),slider_width=0.02,title_height=0.02)

    plotter.add_slider_widget(lambda value: slider("d_spacing",value), [0.5, 10],value=d_spacing_init,title="Spacing", fmt="%1.1f",pointa=(0.01,.9),pointb=(0.2,.9),slider_width=0.02,title_height=0.02)




    plotter.show(full_screen=False)


Helix.plot_curvature = plot_curvature






def transfo_inv(Pos_transfo,D,P,slice_thickness=5):
    """
    This function computes the inverse transformation of a list of transformed position
    """

    #Because of the sliding due to the issue with periodicity
    #The basis of the helix is often not at z = 0
    Pos_slice = Pos_transfo[Pos_transfo[:,2] < (slice_thickness + np.min(Pos_transfo[:,2]))]
    mean_slice = np.mean(Pos_slice)


    mean = np.mean(Pos_transfo,axis=0)
    # print(mean)
    mean[2] = 0
    u,v,w = (Pos_transfo - mean).transpose()
    # u,v,w = (Pos_transfo).transpose()

    # u,v,w = (Pos_transfo).transpose()
    #if unlucky and a point has u=0. Should not happen because if would mean that there are no holes inside
    u = u + (u==0) * 0.00001
    P_t = P/2/np.pi
    R = D
    print('R',R)
    # R = 29.133407802862774
    # P_t = 32.677991490301196
    # R=40




    Lim = R/P_t**2 * (v**2+u**2)**(1/2)
    Lim_m = w/P_t - Lim
    Lim_M = w/P_t + Lim
    Nz = 500
    z = np.linspace(Lim_m,Lim_M, Nz)
    # z = z%(2*np.pi)



    #z is found by finding the roots of this equation
    equation_z = abs(P_t*z - R/P_t * (np.sign(u) * (v**2+u**2)**(1/2) * np.cos((z + np.arctan(v/u)))) - w)
    roots_z_ind = []
    for curv in equation_z.transpose():
        ind_mins = np.array(sps.argrelmin(curv,mode="wrap")[0])
        values_mins = curv[ind_mins]

        # ind_mins = ind_mins[values_mins<0.5]
        #
        # if len(ind_mins)==0:
        #     plt.plot(curv,"or")
        #     plt.show()

        ind_mins_center = abs(ind_mins-Nz//2)
        # print(ind_mins,ind_mins_center)
        roots_z_ind_curv = ind_mins[np.argmin(ind_mins_center)]
        roots_z_ind.append(roots_z_ind_curv)

    roots_z_ind = np.array(roots_z_ind)



    # equation_z = abs(P_t*z + R/P_t * (np.sin(z)*v - np.cos(z)*u)  - w)
    # roots_z_ind = np.argmin(equation_z,axis=0)
    # print(np.shape(roots_z_ind))
    # print("Max err",np.max(np.min(equation_z,axis=0)))


    roots_z = np.take_along_axis(z,np.expand_dims(roots_z_ind,0),axis=0)[0]
    print("Max err",np.max(abs(P_t*roots_z - R/P_t * (np.sign(u) * (v**2+u**2)**(1/2) * np.cos((roots_z + np.arctan(v/u)))) - w)))

    # roots_z = roots_z%(2*np.pi)
    # roots_z = roots_z + np.min(roots_z) + np.pi

    N = (R**2 + P_t**2)**(1/2)
    print("N",N)
    print("P_t",P_t)

    # print(np.min(z),np.max(z))
    x = N/P_t * (-np.cos(roots_z)*u + np.sin(roots_z)*v)
    y = R - np.sin(roots_z)*u - np.cos(roots_z)*v
    # y = R - np.sin(roots_z)*u/2 - np.cos(roots_z)*v/2

    # print(np.max(u),np.max(v))
    # print(np.max(x),np.max(y))

    d = np.argmax(x)
    # print(np.shape(equation_z))
    # plt.plot(z[:,d],equation_z[:,d],"or")
    # plt.show()

    return x,y,roots_z * P_t





def evaluate_dimenions(Pos_transfo,Types,slice_thickness=5,threshold_cut=0.01,threshold_lr=2,show=False):
    def Rz(alpha):
        """3D Rotation matrix around the z axis"""
        return np.array([[np.cos(alpha),-np.sin(alpha),0],[np.sin(alpha),np.cos(alpha),0],[0,0,1]])
    def Ry(beta):
        """3D Rotation matrix around the y axis"""
        return np.array([[np.cos(beta),0,np.sin(beta)],[0,1,0],[-np.sin(beta),0,np.cos(beta)]])
    def Rx(gamma):
        """3D Rotation matrix around the x axis"""
        return np.array([[1,0,0],[0,np.cos(gamma),-np.sin(gamma)],[0,np.sin(gamma),np.cos(gamma)]])
    def Rota(alpha):
        return np.array([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]])



    def calc_dist(Pos,xa,ya,xb,yb):
        """Compute the distance between positions and the segment drawn by two points"""
        Vec = np.array([xb-xa,yb-ya])

        Vec_d_a = Pos - np.array([xa,ya])
        Vec_d_b = Pos - np.array([xb,yb])

        D_a = np.linalg.norm(Vec_d_a,axis=1)
        D_b = np.linalg.norm(Vec_d_b,axis=1)

        Vec = Vec.reshape((1,2))

        D_a_sup_b = (D_a > D_b).reshape((len(D_a),1))
        D_b_sup_a = (D_a <= D_b).reshape((len(D_b),1))

        Vec_calc = Vec * D_b_sup_a - Vec * D_a_sup_b
        Vec_d_calc = Vec_d_a * D_b_sup_a + Vec_d_b * D_a_sup_b

        #Compute the angle to know if the orthogonal of the line containing the Pos is in the segment or not
        cos_angle = np.sum(Vec_d_calc * Vec_calc,axis=1) / np.linalg.norm(Vec_d_calc,axis=1) / np.linalg.norm(Vec_calc,axis=1)
        cos_angle = (cos_angle > 1) * 1 - (cos_angle < -1) * 1 + cos_angle * (cos_angle <= 1) * (cos_angle >= -1)
        angle = np.arccos(cos_angle)

        Dist = (angle < np.pi/2) * np.linalg.norm(Vec_d_calc,axis=1) * abs(np.sin(angle)) + (angle >= np.pi/2) * np.min(np.array([D_a,D_b]),axis=0)


        return Dist

    ##Compute the pitch as the difference between max z - min z
    N_points = len(Pos_transfo)
    min_z = np.min(Pos_transfo[:,2])
    pitch = np.max(Pos_transfo[:,2]) - np.min(Pos_transfo[:,2]) #Not the perfect formula as empty space, but close enough
    N_z = int((pitch // slice_thickness)+1)


    if False:
        #Compute the diameter through the max distance of the flattened helix
        Pos_centered = Pos_transfo - np.mean(Pos_transfo,axis=0)
        Pos_centered[:,2] = 0
        Dist = np.sort(np.sum(Pos_centered**2,axis=1)**(1/2))
        Dist = Dist[int(threshold_cut * N_points):int(1-threshold_cut * N_points)]
        diameter = np.max(Dist)*2
        thickness = np.max(Dist) - np.min(Dist)


        #Compute the diameters using a linear regression
        a,b = np.polyfit(np.arange(len(Dist)),Dist,1)
        diameter = (a*N_points+b) * 2
        thickness = a*N_points


    ##Recenter the positions
    mean = np.mean(Pos_transfo,axis=0)
    mean[2] = 0
    Pos_transfo = Pos_transfo - mean
    Means = []

    ##Compute the factor D as the average radius with the center
    for slice_helix in range(N_z):
        Pos_slice = Pos_transfo[(Pos_transfo[:,2] >= slice_helix*slice_thickness + min_z) * (Pos_transfo[:,2] < (slice_helix+1)*slice_thickness + min_z)]
        radius = np.sum(Pos_slice[:,:2]**2,axis=1)**(1/2)
        mean = np.mean(radius,axis=0)
        Means.append(np.sum(mean**2)**(1/2))


    D = np.mean(Means)

    ##Do the invert transformation to get the slices correctly for the width
    x,y,z = transfo_inv(Pos_transfo,D,pitch)
    Pos = np.array([x,y,z]).transpose()

    center = np.mean(Pos_transfo,axis=0)


    #Do translation to have whole slices using periodic conditions
    Pos_under_z = (Pos[:,2] < min_z).reshape((N_points,1)) * np.array([1.,1.,1.]) * (Pos + np.array([0,0,pitch]))
    Pos_over_z = (Pos[:,2] > (min_z+pitch)).reshape((N_points,1)) * np.array([1.,1.,1.]) * (Pos - np.array([0,0,pitch]))
    Pos_middle_z = ((Pos[:,2] >= min_z) * (Pos[:,2] <= (min_z + pitch))).reshape((N_points,1)) * np.array([1.,1.,1.]) * (Pos)

    Pos_transfo_under_z = (Pos[:,2] < min_z).reshape((N_points,1)) * np.array([1.,1.,1.]) * (Pos_transfo + np.array([0,0,pitch]))
    Pos_transfo_over_z = (Pos[:,2] > (min_z+pitch)).reshape((N_points,1)) * np.array([1.,1.,1.]) * (Pos_transfo - np.array([0,0,pitch]))
    Pos_transfo_middle_z = ((Pos[:,2] >= min_z) * (Pos[:,2] <= (min_z + pitch))).reshape((N_points,1)) * np.array([1.,1.,1.]) * (Pos_transfo)

    Pos_transfo = Pos_transfo_under_z + Pos_transfo_middle_z + Pos_transfo_over_z
    Pos = Pos_under_z + Pos_middle_z + Pos_over_z
    x,y,z = Pos.transpose()


    W_int, W_ext = [],[]
    T_int, T_ext = [],[]
    W = []
    Surf = []
    Surf_tot = []
    T = []

    for slice_helix in range(N_z):
    # def slide(value):
    #     slice_helix = int(value)
        ##Get slices and center them
        Pos_slice = Pos_transfo[(Pos[:,2] >= slice_helix*slice_thickness + min_z) * (Pos[:,2] < (slice_helix+1)*slice_thickness + min_z)]
        Diff_slice_mean = center - np.mean(Pos_slice,axis=0)
        Pos_slice = Pos_slice - np.mean(Pos_slice,axis=0)

        ## rotate to face the same direction
        angle_z = np.arctan2(Diff_slice_mean[1],Diff_slice_mean[0])
        Pos_slice = Pos_slice.dot(Rz(angle_z))

        Num_S = len(Pos_slice)

        ##get the surface to rotate to be on the xy plane
        def plane_surf(X,a,b):
            x,y = X.transpose()
            return a*x + b*y

        a,b = sp.optimize.curve_fit(plane_surf,Pos_slice[:,:2],Pos_slice[:,2])[0]

        Vec_1 = np.array([1,0,plane_surf(np.array([1,0]),a,b)])
        ar = pv.Arrow([0,0,0],Vec_1,scale=10)

        Vec_2 = np.array([0,1,plane_surf(np.array([0,1]),a,b)])
        ar = pv.Arrow([0,0,0],Vec_2,scale=10)
        angle_y = np.arctan2(Vec_1[0],Vec_1[2])
        Pos_slice = Pos_slice.dot(Ry(angle_y-np.pi/2))

        angle_x = np.arctan2(Vec_2[2],Vec_2[1])
        Pos_slice = Pos_slice.dot(Rx(angle_x))


        angles = np.arctan2(Pos_slice[:,1],Pos_slice[:,0]) + np.pi
        dist = np.linalg.norm(Pos_slice,axis=1)





        def square_fit_T(Pos,l,L,phi):
            Pos_corners = np.array([[-l/2, -L/2],[l/2,-L/2],[l/2,L/2],[-l/2,L/2]])
            Pos_corners_rota = Pos_corners.dot(Rota(phi))
            x1,y1,x2,y2,x3,y3,x4,y4 = Pos_corners_rota.ravel()

            Dist_1 = calc_dist(Pos,x1,y1,x2,y2)
            Dist_2 = calc_dist(Pos,x2,y2,x3,y3)
            Dist_3 = calc_dist(Pos,x3,y3,x4,y4)
            Dist_4 = calc_dist(Pos,x4,y4,x1,y1)

            Dists = np.array([Dist_1,Dist_2,Dist_3,Dist_4])
            Min_dists = np.min(Dists,axis=0)
            return Min_dists


        angles = np.arctan2(Pos_slice[:,1],Pos_slice[:,0])+np.pi
        N_angles=64
        d_angle = 2*np.pi/N_angles
        mean_pos_angle=[]
        angles_select_int = np.linspace(0,2*np.pi,N_angles,endpoint=False)
        for angle in angles:
            selection = (angles >= angle) * (angles < (angle + d_angle))
            mean_pos_angle.append(np.mean(Pos_slice[selection],axis=0))
        mean_pos_angle = np.array(mean_pos_angle)

        x_min,x_max,y_min,y_max = np.min(Pos_slice[:,0]),np.max(Pos_slice[:,0]),np.min(Pos_slice[:,1]),np.max(Pos_slice[:,1])
        l,L,phi = x_max-x_min,y_max-y_min,0
        Bounds = np.array([[0,l],[0,L],[0,np.pi*2]]).transpose()

        l,L,phi = sp.optimize.curve_fit(square_fit_T,mean_pos_angle[:,:2],[0]*len(Pos_slice),p0=(l,L,phi),bounds=Bounds)[0]


        Pos_slice = Pos_slice.dot(Rz(-phi))


        Pos_x = Pos_slice[:,0][int(Num_S*threshold_cut):int(Num_S*(1-threshold_cut))]
        width = np.max(Pos_slice[:,1]) - np.min(Pos_slice[:,1])
        W.append(width)
        T.append(np.max(Pos_slice[:,0]) - np.min(Pos_slice[:,0]))





        Pos_slice = Pos_slice[:,:2]



        if False:
            def points(Pos,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8):

                Dist_1 = calc_dist(Pos,x1,y1,x2,y2)
                Dist_2 = calc_dist(Pos,x2,y2,x3,y3)
                Dist_3 = calc_dist(Pos,x3,y3,x4,y4)
                Dist_4 = calc_dist(Pos,x4,y4,x5,y5)
                Dist_5 = calc_dist(Pos,x5,y5,x6,y6)
                Dist_6 = calc_dist(Pos,x6,y6,x7,y7)
                Dist_7 = calc_dist(Pos,x7,y7,x8,y8)
                Dist_8 = calc_dist(Pos,x8,y8,x1,y1)

                Dists = np.array([Dist_1,Dist_2,Dist_3,Dist_4,Dist_5,Dist_6,Dist_7,Dist_8])
                Min_dists = np.min(Dists,axis=0)
                return Min_dists

            def points_arg(Pos,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8):

                Dist_1 = calc_dist(Pos,x1,y1,x2,y2)
                Dist_2 = calc_dist(Pos,x2,y2,x3,y3)
                Dist_3 = calc_dist(Pos,x3,y3,x4,y4)
                Dist_4 = calc_dist(Pos,x4,y4,x5,y5)
                Dist_5 = calc_dist(Pos,x5,y5,x6,y6)
                Dist_6 = calc_dist(Pos,x6,y6,x7,y7)
                Dist_7 = calc_dist(Pos,x7,y7,x8,y8)
                Dist_8 = calc_dist(Pos,x8,y8,x1,y1)

                Dists = np.array([Dist_1,Dist_2,Dist_3,Dist_4,Dist_5,Dist_6,Dist_7,Dist_8])
                arg_dists = np.argmin(Dists,axis=0)
                return arg_dists

            x_min,x_max,y_min,y_max = np.min(Pos_slice[:,0]),np.max(Pos_slice[:,0]),np.min(Pos_slice[:,1]),np.max(Pos_slice[:,1])
            d = 3
            dx = (x_max - x_min)/d
            dy = (y_max - y_min)/d
            x1,y1,x2,y2 = x_min,y_min+dy, x_min+dx, y_min
            x3,y3,x4,y4 = x_max-dx,y_min, x_max, y_min+dy
            x5,y5,x6,y6 = x_max,y_max-dy, x_max-dx,y_max
            x7,y7,x8,y8 = x_min+dx,y_max, x_min,y_max-dy
            p0 = (x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8)
            bounds_x_m = (x_min,0)
            bounds_x_M = (0,x_max)

            bounds_y_m = (y_min,0)
            bounds_y_M = (0,y_max)

            Bounds = np.array( [bounds_x_m,bounds_y_m,bounds_x_m,bounds_y_m,bounds_x_M,bounds_y_m,bounds_x_M,bounds_y_m,bounds_x_M,bounds_y_M,bounds_x_M,bounds_y_M,bounds_x_m,bounds_y_M,bounds_x_m,bounds_y_M]).transpose()


            angles = np.arctan2(Pos_slice[:,1],Pos_slice[:,0]) + np.pi
            N_angles=64
            d_angle = 2*np.pi/N_angles
            mean_pos_angle=[]
            angles_select_int = np.linspace(0,2*np.pi,N_angles,endpoint=False)
            for angle in angles:
                selection = (angles >= angle) * (angles < (angle + d_angle))
                mean_pos_angle.append(np.mean(Pos_slice[selection],axis=0))
            mean_pos_angle = np.array(mean_pos_angle)



            x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8 = sp.optimize.curve_fit(points,mean_pos_angle[:,:2],[0]*len(Pos_slice),p0=p0,bounds=Bounds)[0]

            Vecs_ext = np.array([[x3-x4,y3-y4],[x5-x4,y5-y4],[x5-x6,y5-y6]])
            Vecs_int = np.array([[x7-x8,y7-y8],[x8-x1,y8-y1],[x1-x2,y1-y2]])

            arg_dists = points_arg(Pos_slice[:,:2],x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8)

            ext_points = Pos_slice[((arg_dists == 3) + (arg_dists == 4) + (arg_dists == 5)).astype("bool")][:,:2]
            dists_ext = points(ext_points,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8)
            thickness_ext = np.sort(dists_ext)
            thickness_ext = thickness_ext[:int((1-threshold_cut)*len(thickness_ext))][-1]

            int_points = Pos_slice[((arg_dists == 1) + (arg_dists == 7) + (arg_dists == 8)).astype("bool")][:,:2]
            dists_int = points(int_points,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8)
            thickness_int = np.sort(dists_int)
            thickness_int = thickness_int[:int((1-threshold_cut)*len(thickness_int))][-1]

            # T_ext.append(thickness_ext*2)
            # T_int.append(thickness_int*2)
            # W_ext.append(np.sum(np.linalg.norm(Vecs_ext,axis=1)))
            # W_int.append(np.sum(np.linalg.norm(Vecs_int,axis=1)))


            # print("t_ext",thickness_ext*2)
            # print("t_int",thickness_int*2)
            # print("w_ext",np.sum(np.linalg.norm(Vecs_ext,axis=1)))
            # print("w_int",np.sum(np.linalg.norm(Vecs_int,axis=1)))

        # plt.plot(Pos_slice[:,0],Pos_slice[:,1],"or")
        # plt.show()
        # c=d
            # print(np.array([x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8]))

        N_y = 20
        if True:

            y_int = np.linspace(y_min,y_max,N_y,endpoint=False)
            dy = y_int[1] - y_int[0]
            L_T_int = []
            for y in y_int:
                Pos_extr = Pos_slice[(Pos_slice[:,0] > 0) * (Pos_slice[:,1] >= y) * (Pos_slice[:,1] <= (y + dy))]
                L_T_int.append( np.max(Pos_extr[:,0]) - np.min(Pos_extr[:,0]))
            T_int.append(np.median(L_T_int))



            y_ext = np.linspace(y_min,y_max,N_y,endpoint=False)
            dy = y_ext[1] - y_ext[0]
            L_T_ext = []
            for y in y_ext:
                Pos_extr = Pos_slice[(Pos_slice[:,0] < 0) * (Pos_slice[:,1] >= y) * (Pos_slice[:,1] <= (y + dy))]
                L_T_ext.append( np.max(Pos_extr[:,0]) - np.min(Pos_extr[:,0]))
            T_ext.append(np.median(L_T_ext))


        ##FUNC IMG
        if False:
            # Pos_slice = Pos[(Pos[:,2] >= slice_helix*slice_thickness + min_z) * (Pos[:,2] < (slice_helix+1)*slice_thickness + min_z)]
            # mean_slice = np.mean(Pos_slice,axis=0)
            # Pos_slice = (Pos_slice - mean_slice).dot(Rz(-angle_x))
            Pos_slice = Pos_transfo[(Pos[:,2] >= slice_helix*slice_thickness + min_z) * (Pos[:,2] < (slice_helix+1)*slice_thickness + min_z)]

            Diff_slice_mean = center - np.mean(Pos_slice,axis=0)
            Pos_slice = Pos_slice - np.mean(Pos_slice,axis=0)

            angle_z = np.arctan2(Diff_slice_mean[1],Diff_slice_mean[0])

            Pos_slice = Pos_slice.dot(Rz(angle_z))

            mean_slice = np.mean(Pos_slice,axis=0)
            Num_S = len(Pos_slice)
            Pos_slice = Pos_slice - mean_slice


            def plane_surf(X,a,b):
                x,y = X.transpose()
                return a*x + b*y

            a,b = sp.optimize.curve_fit(plane_surf,Pos_slice[:,:2],Pos_slice[:,2])[0]

            Vec_1 = np.array([1,0,plane_surf(np.array([1,0]),a,b)])
            Vec_2 = np.array([0,1,plane_surf(np.array([0,1]),a,b)])

            if False:
                ar = pv.Arrow([0,0,0],Vec_1,scale=10)
                plotter.add_mesh(ar,name="ar1",color="red")
                ar = pv.Arrow([0,0,0],Vec_2,scale=10)
                plotter.add_mesh(ar,name="ar2",color="blue")
            if True:
                ar = pv.Arrow([0,0,0],[1,0,0],scale=10)
                plotter.add_mesh(ar,name="ar3",color="Black")
                ar = pv.Arrow([0,0,0],[0,1,0],scale=10)
                plotter.add_mesh(ar,name="ar4",color="Black")
                ar = pv.Arrow([0,0,0],[0,0,1],scale=10)
                plotter.add_mesh(ar,name="ar5",color="Black")

            angle_y = np.arctan2(Vec_1[0],Vec_1[2])
            Pos_slice = Pos_slice.dot(Ry(angle_y-np.pi/2))

            angle_x = np.arctan2(Vec_2[2],Vec_2[1])
            Pos_slice = Pos_slice.dot(Rx(angle_x))


            # angles = np.arctan2(Pos_slice[:,1],Pos_slice[:,0]) + np.pi
            # dist = np.linalg.norm(Pos_slice,axis=1)
            #
            # def coss(x,a,phi,b,w):
            #     return -a*np.cos(w*x+phi) + b
            # a, phi, b, w = (np.max(dist)-np.min(dist)) /2, 0, np.mean(dist), 2
            # a, phi, b, w = sp.optimize.curve_fit(coss,angles,dist,p0=(a,phi,b,w))[0]
            # # Pos_slice = Pos_slice.dot(Rz(-phi+np.pi/2))
            # print("PHI",phi)
            # x = np.linspace(0,2*np.pi,100)
            # plt.plot(x,coss(x,a,phi,b,w),"o-r")
            # print("PHI_av",phi)
            #
            # angles = np.arctan2(Pos_slice[:,1],Pos_slice[:,0]) + np.pi
            # dist = np.linalg.norm(Pos_slice,axis=1)
            # a, phi, b, w = (np.max(dist)-np.min(dist)) /2, 0, np.mean(dist), 2
            # a, phi, b, w = sp.optimize.curve_fit(coss,angles,dist,p0=(a,phi,b,w))[0]
            # print("PHI_ap",phi)
            # plt.plot(x,coss(x,a,phi,b,w),"o-b")
            # plt.show()

            def points(Pos,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8):
                X,Y = Pos.transpose()


                Dist_1 = calc_dist(Pos,x1,y1,x2,y2)
                Dist_2 = calc_dist(Pos,x2,y2,x3,y3)
                Dist_3 = calc_dist(Pos,x3,y3,x4,y4)
                Dist_4 = calc_dist(Pos,x4,y4,x5,y5)
                Dist_5 = calc_dist(Pos,x5,y5,x6,y6)
                Dist_6 = calc_dist(Pos,x6,y6,x7,y7)
                Dist_7 = calc_dist(Pos,x7,y7,x8,y8)
                Dist_8 = calc_dist(Pos,x8,y8,x1,y1)

                Dists = np.array([Dist_1,Dist_2,Dist_3,Dist_4,Dist_5,Dist_6,Dist_7,Dist_8])
                Min_dists = np.min(Dists,axis=0)
                return Min_dists

            def square_fit(Pos,x1,y1,x2,y2,x3,y3,x4,y4):

                Dist_1 = calc_dist(Pos,x1,y1,x2,y2)
                Dist_2 = calc_dist(Pos,x2,y2,x3,y3)
                Dist_3 = calc_dist(Pos,x3,y3,x4,y4)
                Dist_4 = calc_dist(Pos,x4,y4,x1,y1)

                Dists = np.array([Dist_1,Dist_2,Dist_3,Dist_4])
                Min_dists = np.min(Dists,axis=0)
                return Min_dists




            def square_fit_T(Pos,l,L,phi):
                Pos_corners = np.array([[-l/2, -L/2],[l/2,-L/2],[l/2,L/2],[-l/2,L/2]])
                Pos_corners_rota = Pos_corners.dot(Rota(phi))
                x1,y1,x2,y2,x3,y3,x4,y4 = Pos_corners_rota.ravel()

                Dist_1 = calc_dist(Pos,x1,y1,x2,y2)
                Dist_2 = calc_dist(Pos,x2,y2,x3,y3)
                Dist_3 = calc_dist(Pos,x3,y3,x4,y4)
                Dist_4 = calc_dist(Pos,x4,y4,x1,y1)

                Dists = np.array([Dist_1,Dist_2,Dist_3,Dist_4])
                Min_dists = np.min(Dists,axis=0)
                return Min_dists


            angles = np.arctan2(Pos_slice[:,1],Pos_slice[:,0])+np.pi
            N_angles=64
            d_angle = 2*np.pi/N_angles
            mean_pos_angle=[]
            angles_select_int = np.linspace(0,2*np.pi,N_angles,endpoint=False)
            for angle in angles:
                selection = (angles >= angle) * (angles < (angle + d_angle))
                mean_pos_angle.append(np.mean(Pos_slice[selection],axis=0))
            mean_pos_angle = np.array(mean_pos_angle)

            x_min,x_max,y_min,y_max = np.min(Pos_slice[:,0]),np.max(Pos_slice[:,0]),np.min(Pos_slice[:,1]),np.max(Pos_slice[:,1])
            l,L,phi = x_max-x_min,y_max-y_min,0
            Bounds = np.array([[0,l],[0,L],[0,np.pi*2]]).transpose()
            print(l,L,phi)
            l,L,phi = sp.optimize.curve_fit(square_fit_T,mean_pos_angle[:,:2],[0]*len(Pos_slice),p0=(l,L,phi),bounds=Bounds)[0]
            print(l,L,phi)

            Pos_slice = Pos_slice.dot(Rz(-phi))


            if False:
                x_min,x_max,y_min,y_max = np.min(Pos_slice[:,0]),np.max(Pos_slice[:,0]),np.min(Pos_slice[:,1]),np.max(Pos_slice[:,1])
                d = 3
                dx = (x_max - x_min)/d
                dy = (y_max - y_min)/d
                x1,y1,x2,y2 = x_min,y_min+dy, x_min+dx, y_min
                x3,y3,x4,y4 = x_max-dx,y_min, x_max, y_min+dy
                x5,y5,x6,y6 = x_max,y_max-dy, x_max-dx,y_max
                x7,y7,x8,y8 = x_min+dx,y_max, x_min,y_max-dy
                p0 = (x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8)
                bounds_x_m = (x_min,0)
                bounds_x_M = (0,x_max)

                bounds_y_m = (y_min,0)
                bounds_y_M = (0,y_max)

                Bounds = np.array( [bounds_x_m,bounds_y_m,bounds_x_m,bounds_y_m,bounds_x_M,bounds_y_m,bounds_x_M,bounds_y_m,bounds_x_M,bounds_y_M,bounds_x_M,bounds_y_M,bounds_x_m,bounds_y_M,bounds_x_m,bounds_y_M]).transpose()


                angles = np.arctan2(Pos_slice[:,1],Pos_slice[:,0]) + np.pi
                N_angles=64
                d_angle = 2*np.pi/N_angles
                mean_pos_angle=[]
                angles_select_int = np.linspace(0,2*np.pi,N_angles,endpoint=False)
                for angle in angles:
                    selection = (angles >= angle) * (angles < (angle + d_angle))
                    mean_pos_angle.append(np.mean(Pos_slice[selection],axis=0))
                mean_pos_angle = np.array(mean_pos_angle)



                x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8 = sp.optimize.curve_fit(points,mean_pos_angle[:,:2],[0]*len(Pos_slice),p0=p0,bounds=Bounds)[0]


                # print(np.array([x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8]))

            if False:
                angles = np.arctan2(Pos_slice[:,1],Pos_slice[:,0]) + np.pi
                dist = np.linalg.norm(Pos_slice,axis=1)

                def coss(x,a,phi,b,w):
                    return a*np.cos(w*x+phi) + b
                a, phi, b, w = (np.max(dist)-np.min(dist)) /2, angles[np.argmax(dist)], np.mean(dist), 2
                print(a,phi,b,w)
                a,phi,b,w = sp.optimize.curve_fit(coss,angles,dist,p0=(a,phi,b,w))[0]
                # a,phi,b,w = sp.optimize.curve_fit(coss,angle,dist)[0]
                print(a, phi, b, w)
                if a < 0: a, phi = -a, phi + np.pi

                x=np.linspace(0,2*np.pi,100)
                plt.plot(x,coss(x,a,phi,b,w),"-b")
                plt.plot(angles,dist,"or")

                angle_max_1 = (-phi/w) % (2*np.pi)
                angle_max_2 = ((-phi+2*np.pi)/w)% (2*np.pi)

                angle_max_1, angle_max_2 = max(angle_max_1, angle_max_2), min(angle_max_1, angle_max_2)

                plt.vlines(angle_max_1,b-a,a+b,"g")
                plt.vlines(angle_max_2,b-a,a+b,"b")

                select_int = (angles<angle_max_1) * (angles>angle_max_2)
                int_slice = Pos_slice[select_int]
                ext_slice = Pos_slice[(1-select_int).astype("bool")]

                N_angles = 16
                angles_int = angles[select_int]
                min_angle_int, max_angle_int = np.min(angles_int),np.max(angles_int)


                d_angle = (max_angle_int - min_angle_int)/N_angles
                angles_select_int = np.linspace(min_angle_int,max_angle_int,N_angles,endpoint=False)

                thickness_int = []
                mean_pos_angle_int = []
                for angle in angles_select_int:
                    selection = (angles[select_int] >= angle) * (angles[select_int] < (angle + d_angle))

                    int_slice_angle = int_slice[selection]
                    mean_pos_angle_int.append(np.mean(int_slice_angle,axis=0))
                    print(np.sum(selection))
                    thickness_int.append([])

                plt.plot(angles_select_int,coss(angles_select_int,a,phi,b,w),"oy")
                plt.show()

            if False:
                x_min,x_max,y_min,y_max = np.min(Pos_slice[:,0]),np.max(Pos_slice[:,0]),np.min(Pos_slice[:,1]),np.max(Pos_slice[:,1])

                x1,y1,x2,y2,x3,y3,x4,y4 = x_min,y_min, x_max, y_min, x_max,y_max,x_min,y_max

                p0 = (x1,y1,x2,y2,x3,y3,x4,y4)
                bounds_x_m = (x_min,0)
                bounds_x_M = (0,x_max)

                bounds_y_m = (y_min,0)
                bounds_y_M = (0,y_max)

                Bounds = np.array( [bounds_x_m,bounds_y_m,bounds_x_M,bounds_y_m,bounds_x_M,bounds_y_M,bounds_x_m,bounds_y_M]).transpose()
                angles = np.arctan2(Pos_slice[:,1],Pos_slice[:,0]) + np.pi
                N_angles=64
                d_angle = 2*np.pi/N_angles

                mean_pos_angle=[]
                angles_select_int = np.linspace(0,2*np.pi,N_angles,endpoint=False)
                for angle in angles:
                    selection = (angles >= angle) * (angles < (angle + d_angle))
                    mean_pos_angle.append(np.mean(Pos_slice[selection],axis=0))
                mean_pos_angle = np.array(mean_pos_angle)
                # x1,y1,x2,y2,x3,y3,x4,y4 = sp.optimize.curve_fit(square_fit,mean_pos_angle[:,:2],[0]*len(Pos_slice),p0=p0,bounds=Bounds)[0]

                N_y = 20

                # y_int = np.linspace(y_min,y_max,N_y,endpoint=False)
                y_int = np.linspace(y_min,y_max,N_y,endpoint=False)
                dy = y_int[1] - y_int[0]
                L_T_int = []
                for y in y_int:
                    Pos_extr = Pos_slice[(Pos_slice[:,0] > 0) * (Pos_slice[:,1] >= y) * (Pos_slice[:,1] <= (y + dy))]
                    L_T_int.append( np.max(Pos_extr[:,0]) - np.min(Pos_extr[:,0]))
                T_int.append(np.median(L_T_int))


                # y_ext = np.linspace(y1,y4,N_y,endpoint=False)
                y_ext = np.linspace(y_min,y_max,N_y,endpoint=False)
                dy = y_ext[1] - y_ext[0]
                L_T_ext = []
                for y in y_ext:
                    Pos_extr = Pos_slice[(Pos_slice[:,0] < 0) * (Pos_slice[:,1] >= y) * (Pos_slice[:,1] <= (y + dy))]
                    L_T_ext.append( np.max(Pos_extr[:,0]) - np.min(Pos_extr[:,0]))
                T_ext.append(np.median(L_T_ext))











            if False:

                # print(x1,y1,x2,y2,x3,y3,x4,y4)

                tube = pv.Tube(pointa = [x1,y1,0], pointb = [x2,y2,0])
                plotter.add_mesh(tube,name="ta")
                tube = pv.Tube(pointa = [x2,y2,0], pointb = [x3,y3,0])
                plotter.add_mesh(tube,name="tb")
                tube = pv.Tube(pointa = [x3,y3,0], pointb = [x4,y4,0])
                plotter.add_mesh(tube,name="tc")
                tube = pv.Tube(pointa = [x4,y4,0], pointb = [x5,y5,0])
                plotter.add_mesh(tube,name="td")
                tube = pv.Tube(pointa = [x5,y5,0], pointb = [x6,y6,0])
                plotter.add_mesh(tube,name="te")
                tube = pv.Tube(pointa = [x6,y6,0], pointb = [x7,y7,0])
                plotter.add_mesh(tube,name="tf")
                tube = pv.Tube(pointa = [x7,y7,0], pointb = [x8,y8,0])
                plotter.add_mesh(tube,name="tg")
                tube = pv.Tube(pointa = [x8,y8,0], pointb = [x1,y1,0])
                plotter.add_mesh(tube,name="th")


            if False:
                Pos_corners = np.array([[-l/2, -L/2],[l/2,-L/2],[l/2,L/2],[-l/2,L/2]])
                Pos_corners_rota = Pos_corners.dot(Rota(phi))
                x1,y1,x2,y2,x3,y3,x4,y4 = Pos_corners_rota.ravel()
                # print(x1,y1,x2,y2,x3,y3,x4,y4)

                tube = pv.Tube(pointa = [x1,y1,0], pointb = [x2,y2,0])
                plotter.add_mesh(tube,name="ta")
                tube = pv.Tube(pointa = [x2,y2,0], pointb = [x3,y3,0])
                plotter.add_mesh(tube,name="tb")
                tube = pv.Tube(pointa = [x3,y3,0], pointb = [x4,y4,0])
                plotter.add_mesh(tube,name="tc")
                tube = pv.Tube(pointa = [x4,y4,0], pointb = [x1,y1,0])
                plotter.add_mesh(tube,name="td")





            Types_slice = Types[(Pos[:,2] >= slice_helix*slice_thickness + min_z) * (Pos[:,2] < (slice_helix+1)*slice_thickness + min_z)]
            Pos_slice = Pos[:]
            Types_slice = Types[:]

            # Pos_slice = Pos_transfo[:]
            # Types_slice = Types[:]

            # Pos_slice = np.append(Pos_slice,[[0]]*len(Pos_slice),axis=1)

            # x,y,z = transfo_inv(Pos_slice,diameter_avg,pitch,thickness)
            # print(diameter_avg,pitch,thickness)
            # print(np.shape(Pos_slice))
            # Pos_slice = np.array([x,y,z]).transpose()
            # print(np.shape(Pos_slice))

            sph = pv.Sphere(radius=0.2)
            if True:
                sph = pv.Sphere(radius=1.0)
                data = pv.PolyData(mean_pos_angle)
                pc = data.glyph(scale=False,geom=sph,orient=False)
                plotter.add_mesh(pc,opacity=0.5,pbr=True,roughness=.5,metallic=.2,color="blue",name="aa")
            if False:
                bis = np.append(bisector_points,[[0]]*len(bisector_points),axis=1)
                data = pv.PolyData(bis)
                pc = data.glyph(scale=False,geom=sph,orient=False)
                plotter.add_mesh(pc,opacity=0.5,pbr=True,roughness=.5,metallic=.2,color="green",name="bis")

            if False:
                for a,b in zip(perpendicular_dir_coef,perpendicular_affin):
                    x = np.linspace(-90,120,10)
                    y = a*x+b
                    z = np.array([0]*len(x))
                    X,Y,Z = np.meshgrid(x,y,z)
                    data =pv.StructuredGrid(x,y,z)
                    plotter.add_mesh(data,opacity=0.5,pbr=True,roughness=.5,metallic=.2,color="red")

            if False:
                sph = pv.Sphere(radius=0.6)
                data = pv.PolyData(int_slice)
                pc = data.glyph(scale=False,geom=sph,orient=False)
                plotter.add_mesh(pc,opacity=0.5,pbr=True,roughness=.5,metallic=.2,color="gray",name="si")

                sph = pv.Sphere(radius=0.4)
                data = pv.PolyData(ext_slice)
                pc = data.glyph(scale=False,geom=sph,orient=False)
                plotter.add_mesh(pc,opacity=0.5,pbr=True,roughness=.5,metallic=.2,color="red",name="o")

            else:


                # plotter=pv.Plotter()
                # plotter.show_axes()
                sph = pv.Sphere(radius=0.6)
                data = pv.PolyData(Pos_slice[Types_slice==1])
                pc = data.glyph(scale=False,geom=sph,orient=False)
                plotter.add_mesh(pc,opacity=0.5,pbr=True,roughness=.5,metallic=.2,color="gray",name="si")

                sph = pv.Sphere(radius=0.4)
                data = pv.PolyData(Pos_slice[(Types_slice==2) + (Types_slice==3)])
                pc = data.glyph(scale=False,geom=sph,orient=False)
                plotter.add_mesh(pc,opacity=0.5,pbr=True,roughness=.5,metallic=.2,color="red",name="o")


            if True:
                sph = pv.Sphere(radius=1.2)
                centers = pv.PolyData([[0.,0,0]])
                pc = centers.glyph(scale=False,geom=sph,orient=False)
                plotter.add_mesh(pc,opacity=1.0,pbr=True,roughness=.5,metallic=.2,color="black",name="center")


            # plotter.show()

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

    if show:
        plotter = pv.Plotter()
        plotter.show_axes()
        plotter.add_slider_widget(slide, [0,N_z-1],value=N_z/2,title="Pos", fmt="%3.3e")
        plotter.show()

    print("W",np.median(W))
    # print("W_int",np.mean(W_int))
    # print("W_ext",np.mean(W_ext))
    print("T_int",np.median(T_int))
    print("T_ext",np.median(T_ext))
    print("T",np.median(T))
    print("pitch",pitch)
    print("thickness",thickness)
    print("D",2*D+np.median(T))
    print("diameter",diameter)
    return pitch, np.median(W), np.mediant(T), 2*D+np.median(T)

def evaluate_surface_reconstruct(Pos_transfo,Types,slice_thickness=5,threshold_cut=0.01,threshold_lr=2):
    rota = 1
    pitch, width, thickness, D = evaluate_dimenions(Pos_transfo,Types,slice_thickness,threshold_cut,threshold_lr)
    Pos_model_no_rota,Types_model,Lims_tot,Angles_OH, Pos_transfo_int, Types_int, Lims_tot_int = create_syst(0,D,pitch,width,thickness,int_thick,do_clean=False,circling=True,do_periodic=False,file_duplicate="grid.data")
    Pos_model,Types_model,Lims_tot,Angles_OH, Pos_transfo_int, Types_int, Lims_tot_int = create_syst(rota,D,pitch,width,thickness,int_thick,do_clean=False,circling=True,do_periodic=False,file_duplicate="grid.data")

    angles = np.arctan2(Pos_model_no_rota[:,1],Pos_model_no_rota[:,0])
    dz = 1
    min_z, max_z = np.min(Pos_model_no_rota[:,2]),np.max(Pos_model_no_rota[:,2])
    Nz = (max_z - min_z)/dz
    Pos = []
    for nz in range(Nz):
        Slice = (Pos_mode_no_rota[:,2] >= (nz * dz)) * (Pos_mode_no_rota[:,2] <= (nz * dz + dz))
        Pos.append(Pos_model_no_rota[Slice])

    Pos = np.array(Pos)

    plotter = pv.Plotter()
    plotter.add_axes()



    Si_c = [240,200,160]
    O_c = [255,13,13]
    O_c2 = [0,255,0]
    H_c = [255,255,255]
    H_c2 = [0,0,255]

    sp = pv.Sphere(radius=0.4)
    Pos_transfo = np.array(Pos)
    Pos_t

    data = pv.PolyData(Pos_transfo[Types==1])
    pc = data.glyph(scale=False,geom=sp,orient=False)
    plotter.add_mesh(pc,opacity=0.9,pbr=True,roughness=.5,metallic=.2,color=Si_c,scalars=np.arange(len(Pos_transfo)*len(sp.points)))


    Lims, Atom_types, Atom_pos = read_data("grid.data",do_scale=False,atom_style="atom")


    for z in range(len(Pos_transfo)-Nz-1):

        # P = [Pos_transfo[z],Pos_transfo[z+1],Pos_transfo[z+Nz],Pos_transfo[z+Nz+1]]
        P = [Pos_transfo[z],Pos_transfo[z+Nz]]
        x_coord,y_coord,z_coord = np.array(P).transpose()
        grid = pv.StructuredGrid(x_coord,y_coord,z_coord)
        plotter.add_mesh(grid,opacity=0.9,pbr=True,roughness=.5,metallic=.2)
