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
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import scipy.spatial.distance as sd
from shanc.read_write import *
import matplotlib.pyplot as plt
import pyvista as pv
from shanc.helix_object import *


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



def compute_bonds_system(helix,cube=50,d_max=5):
    """
    compute_bonds_system(helix,cube=50,d_max=5)

    Compute bond counts and RDF distances by slicing the system into sub-cubes.

    Parameters
    ----------
        helix : Helix
            The helix on which to compute the bonds
        cube : float, optional
            the length parameter of the cubes used to divide the system. By default 50 A
        d_max : float, optional
            the length by which each cube is enlarged to compute bonds with other cubes. By default 5 A

    Returns
    -------
        Bonds
            The Bonds of the system as a list of pairs
    """

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


    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):

                #Extract small cubes to compute the bonds
                mask_p = (
                    (Pos_added[:, 0] >= x*cube + lx - d_max) & (Pos_added[:, 0] < (x+1)*cube + lx + d_max) &
                    (Pos_added[:, 1] >= y*cube + ly - d_max) & (Pos_added[:, 1] < (y+1)*cube + ly + d_max) &
                    (Pos_added[:, 2] >= z*cube + lz - d_max) & (Pos_added[:, 2] < (z+1)*cube + lz + d_max)
                )

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



def compute_rdf(helix,cube=50,rdf_max=5):
    """
    compute_rdf(helix,cube=50,rdf_max=5)

    Computes the RDF distances by slicing the system into sub-cubes.

    Parameters
    ----------
        helix : Helix
        The helix on which to compute the RDF
        cube : float, optional
            the length parameter of the cubes used to divide the system. By default 50 A
        rdf_max : float, optional
            the length by which each cube is enlarged to compute distances with other cubes. By default 5 A

    Returns
    -------
        RDF
            The RDF as a list of list. Each list contains the RDF of the type compared with the other types
    """

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
    Rdf = [ [[] for k in range(max_types)] for i in range(max_types) ]

    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):

                #Extract small cubes to compute the bonds
                mask_p = (
                    (Pos_added[:, 0] >= x*cube + lx - rdf_max) & (Pos_added[:, 0] < (x+1)*cube + lx + rdf_max) &
                    (Pos_added[:, 1] >= y*cube + ly - rdf_max) & (Pos_added[:, 1] < (y+1)*cube + ly + rdf_max) &
                    (Pos_added[:, 2] >= z*cube + lz - rdf_max) & (Pos_added[:, 2] < (z+1)*cube + lz + rdf_max)
                )

                mask_u = (
                    (Pos_added[:, 0] >= x*cube + lx) & (Pos_added[:, 0] < (x+1)*cube + lx) &
                    (Pos_added[:, 1] >= y*cube + ly) & (Pos_added[:, 1] < (y+1)*cube + ly) &
                    (Pos_added[:, 2] >= z*cube + lz) & (Pos_added[:, 2] < (z+1)*cube + lz)
                )
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


    plt.xlabel("Number of bonds", color=purple,size=17)


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

def save_defects(self, file_name="defects.txt",
                 cation_type=1, anion_type=2):
    """
    Re-map atom types to encode coordination defects and write an XYZ file.

    Normal coordination (type 1 → 1, type 2 → 2) is preserved; under- and
    over-coordinated atoms receive higher type ids for easy visualisation.
    """

    print("TO REview")

    Pos = self.Pos
    Types = self.Types
    Lims = self.Lims
    periodic = self.periodic

    _, Cat_count_An, An_count_Cat = compute_rdf(
        Pos, Types, cube=30, periodic=periodic, Lims=Lims, rdf_max=5,
        cation_type=cation_type, anion_type=anion_type,
    )

    # Cation re-mapping: coord 4 → type 1 (normal); others → types 5-12
    for coord, new_type in [(4, 1), (0, 5), (1, 6), (2, 7), (3, 8),
                             (5, 9), (6, 10), (7, 11), (8, 12)]:
        Types[Types == cation_type] = np.where(
            Cat_count_An == coord, new_type, Types[Types == cation_type]
        )

    # Anion re-mapping: coord 2 → type 2 (normal); others → types 13-18
    for coord, new_type in [(2, 2), (0, 13), (1, 14), (3, 15),
                             (4, 16), (5, 17), (6, 18)]:
        Types[Types == anion_type] = np.where(
            An_count_Cat == coord, new_type, Types[Types == anion_type]
        )

    write_xyz(file_name, Pos, Types)

Helix.save_defects = save_defects


# ---------------------------------------------------------------------------
# Surface (QuickSurf) and curvature analysis
# ---------------------------------------------------------------------------

def compute_quick_surface(Pos, grid, Lims, alpha=2, prec=20, d=10,
                           length_box=20, N_th=8):
    """
    Compute a Gaussian-density isosurface (VMD QuickSurf-style).

    This function evaluates a Gaussian density field on a 3D grid by dividing
    the space into smaller boxes and summing contributions from nearby atoms.

    ρ(r) = \\sum exp(−|r − r_i|^2 / 2\\alpha^2)

    The system is divided into boxes to avoid building the full distance
    matrix; multithreading is supported.

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
    import threading as th

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

    def evaluate_surface(cube, Nx_box, Ny_box, Nz_range):
        for z_box in range(Nz_range[0], Nz_range[1]):
            for y_box in range(Ny_box):
                for x_box in range(Nx_box):
                    Ind_Box = (
                        (x >= x_box*length_box + lx - d) & (x <= (x_box+1)*length_box + lx - d) &
                        (y >= y_box*length_box + ly - d) & (y <= (y_box+1)*length_box + ly - d) &
                        (z >= z_box*length_box + lz - d) & (z <= (z_box+1)*length_box + lz - d)
                    )
                    Pos_Box = Pos_Grid[Ind_Box]

                    Ind_trunc = (
                        (Pos[:, 0] >= x_box*length_box - prec + lx - d) & (Pos[:, 0] <= (x_box+1)*length_box + prec + lx - d) &
                        (Pos[:, 1] >= y_box*length_box - prec + ly - d) & (Pos[:, 1] <= (y_box+1)*length_box + prec + ly - d) &
                        (Pos[:, 2] >= z_box*length_box - prec + lz - d) & (Pos[:, 2] <= (z_box+1)*length_box + prec + lz - d)
                    )
                    Pos_trunc = Pos[Ind_trunc]

                    Dist = sd.cdist(Pos_Box, Pos_trunc)
                    cube[Ind_Box] = np.einsum("ij->i", np.exp(-Dist**2 / (2 * alpha**2)))

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


def plot_defects(self, d_spacing=5, isovalue=1., alpha=2., prec=20,
                    d=10, length_box=20, smoothing=1000, N_th=8,
                    cation_type=1, anion_type=2, metal = False):
    """
    Visualise structural defects overlaid on a Gaussian isosurface.

    Defect analysis highlights under- and over-coordinated atoms on a
    QuickSurf-style contour and allows optional cycle drawing.
                    d_spacing=5, isovalue=1., alpha=2., prec=20,
                    d=10, length_box=20, smoothing=1000, N_th=8,
                    cation_type=1, anion_type=2):
    """


    Pos = self.Pos
    Types = self.Types
    Lims = self.Lims
    periodic = self.periodic


    _, Cat_count_An, An_count_Cat = compute_rdf(
        Pos, Types, cube=30, periodic=periodic, Lims=Lims, rdf_max=5,
        cation_type=cation_type, anion_type=anion_type, metal = metal
    )

    plotter = pv.Plotter()
    plotter.add_axes()

    sp = pv.Sphere(radius=0.6)
    Colors_Cat = ["white", "powderblue", "lightsteelblue", "dodgerblue",
                  "blue", "navy", "black"]
    for n_bonds in range(7):
        mask = Cat_count_An == n_bonds
        if not mask.any():
            continue
        if n_bonds == 4 and not metal:
            continue
        data = pv.PolyData(Pos[Types == cation_type][mask])
        plotter.add_mesh(data.glyph(scale=False, geom=sp, orient=False),
                         opacity=1.0, color=Colors_Cat[n_bonds],
                         name=f"Cat_{n_bonds}")

    if not metal:
        anion_mask = (Types == anion_type) | (Types == 3)
        sp = pv.Sphere(radius=0.4)
        Colors_An = ["white", "orange", "red", "darkred", "black"]
        for n_bonds in range(5):
            mask = An_count_Cat == n_bonds
            if n_bonds == 2 or not mask.any():
                continue
            data = pv.PolyData(Pos[anion_mask][mask])
            plotter.add_mesh(data.glyph(scale=False, geom=sp, orient=False),
                            opacity=1.0, color=Colors_An[n_bonds],
                            name=f"An_{n_bonds}")

        Lx, Ly, Lz = np.max(Pos, axis=0) + d
        lx, ly, lz = np.min(Pos, axis=0) - d
        Nx = int(round((Lx - lx + 2*d) / d_spacing)) + 1
        Ny = int(round((Ly - ly + 2*d) / d_spacing)) + 1
        Nz = int(round((Lz - lz + 2*d) / d_spacing)) + 1

        grid    = pv.ImageData(dimensions=(Nx, Ny, Nz),
                            origin=(lx - d, ly - d, lz - d),
                            spacing=(d_spacing, d_spacing, d_spacing))
        cube    = compute_quick_surface(Pos, grid, [[Lx, lx], [Ly, ly], [Lz, lz]],
                                        alpha=alpha, prec=prec, d=d,
                                        length_box=length_box, N_th=N_th)
        contour = grid.contour(isosurfaces=(isovalue,), scalars=cube)
        smooth  = contour.smooth(n_iter=int(smoothing)) if smoothing else contour
        plotter.add_mesh(smooth, opacity=0.1, color="red", name="contour")
        plotter.show()

Helix.plot_defects = plot_defects

def plot_analyze_density(self, d_spacing=5, isovalue=2., alpha=2., prec=20,
                    d=10, length_box=20, smoothing=1000, N_th=8):
    """
    Render an isosurface coloured by local atom density.
    """
    print("TO REview")

    Pos = self.Pos
    Types = self.Types
    Lims = self.Lims
    periodic = self.periodic

    rdf_max = 20
    Lx, Ly, Lz = np.max(Pos, axis=0) + d
    lx, ly, lz = np.min(Pos, axis=0) - d

    Nx = int(round((Lx - lx + 2*d) / d_spacing)) + 1
    Ny = int(round((Ly - ly + 2*d) / d_spacing)) + 1
    Nz = int(round((Lz - lz + 2*d) / d_spacing)) + 1

    grid    = pv.ImageData(dimensions=(Nx, Ny, Nz),
                           origin=(lx - d, ly - d, lz - d),
                           spacing=(d_spacing, d_spacing, d_spacing))
    cube    = compute_quick_surface(Pos, grid, [[Lx, lx], [Ly, ly], [Lz, lz]],
                                    alpha=alpha, prec=prec, d=d,
                                    length_box=length_box, N_th=N_th)
    contour = grid.contour(isosurfaces=(isovalue,), scalars=cube)
    smooth  = contour.smooth(n_iter=int(smoothing)) if smoothing else contour

    Pos_Surface = smooth.points
    Density     = np.array([
        np.sum(
            (Pos[:, 0] > px - rdf_max) & (Pos[:, 0] < px + rdf_max) &
            (Pos[:, 1] > py - rdf_max) & (Pos[:, 1] < py + rdf_max) &
            (Pos[:, 2] > pz - rdf_max) & (Pos[:, 2] < pz + rdf_max)
        )
        for px, py, pz in Pos_Surface
    ], dtype=float)

    Density /= np.max(Density)

    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.add_mesh(smooth, opacity=1.0, cmap="cool", scalars=Density,
                     scalar_bar_args={"title": "Density",
                                      "title_font_size": 40,
                                      "label_font_size": 40})
    plotter.show()

Helix.plot_analyze_density = plot_analyze_density


# ---------------------------------------------------------------------------
# Visualisation – interactive slice viewer
# ---------------------------------------------------------------------------

def analyze_plot_syst(self,
                      draw_limit=5, compute_limit=7,
                      Cycles=None, L_cycles=None,
                      cation_type=1, anion_type=2):
    """
    Plot slices of the system highlighting atoms with wrong coordination.
    A slider lets you navigate along the z-axis of the helix.

    Parameters
    ----------
    Pos, Types     : positions and types
    periodic       : bool  – z-periodic boundary, default False
    Lims           : list  – system limits for PBC
    draw_limit     : float – half-height of drawn slice (Å), default 5
    compute_limit  : float – half-height of computation slice (Å), default 7
    Cycles, L_cycles : optional cycle data from script_cycles
    cation_type    : int   – type id of the cation, default 1
    anion_type     : int   – type id of the anion,  default 2
    """

    Pos = self.Pos
    Types = self.Types
    Lims = self.Lims
    periodic = self.periodic

    def slide(value):
        if periodic:
            L = Lims[2][1]
            Pos_cut_l = (
                ((Pos[:, 2] >= value - compute_limit) & (Pos[:, 2] < value + compute_limit)) |
                ((Pos[:, 2] + L >= value - compute_limit) & (Pos[:, 2] + L < value + compute_limit)) |
                ((Pos[:, 2] - L >= value - compute_limit) & (Pos[:, 2] - L < value + compute_limit))
            )
        else:
            Pos_cut_l = (Pos[:, 2] > value - compute_limit) & (Pos[:, 2] < value + compute_limit)

        Pos_cutted_l   = Pos[Pos_cut_l]
        Types_cutted_l = Types[Pos_cut_l]
        _, Cat_count_An, An_count_Cat, O_count_H, _ = _compute_bonds(
            Pos_cutted_l, Types_cutted_l,
            cation_type=cation_type, anion_type=anion_type,
        )

        Pos_cut    = (Pos_cutted_l[:, 2] > value - draw_limit) & (Pos_cutted_l[:, 2] < value + draw_limit)
        Pos_cutted  = Pos_cutted_l[Pos_cut]
        Types_cutted = Types_cutted_l[Pos_cut]

        select_Cat = Pos_cut[Types_cutted_l == cation_type]
        select_An  = Pos_cut[((Types_cutted_l == anion_type) | (Types_cutted_l == 3))]
        Cat_count_An = Cat_count_An[select_Cat]
        An_count_Cat = An_count_Cat[select_An]
        if O_count_H.any():
            O_count_H_cut = O_count_H[select_An]
        else:
            O_count_H_cut = O_count_H

        sp = pv.Sphere(radius=0.6)
        Colors_Cat = ["white", "powderblue", "lightsteelblue", "dodgerblue",
                      "blue", "navy", "black"]
        for n_bonds in range(7):
            op = 0.05 if n_bonds == 4 else 1.0
            mask = Cat_count_An == n_bonds
            data = pv.PolyData(Pos_cutted[Types_cutted == cation_type][mask]) if mask.any() else pv.PolyData()
            plotter.add_mesh(data.glyph(scale=False, geom=sp, orient=False),
                             opacity=op, color=Colors_Cat[n_bonds],
                             name=f"Cat_{n_bonds}")

        sp = pv.Sphere(radius=0.4)
        Colors_An = ["white", "orange", "red", "darkred", "black"]
        for n_bonds in range(5):
            op = 0.05 if n_bonds == 2 else 1.0
            if O_count_H_cut.any():
                mask = (An_count_Cat + O_count_H_cut) == n_bonds
            else:
                mask = An_count_Cat == n_bonds
            data = (
                pv.PolyData(Pos_cutted[((Types_cutted == anion_type) | (Types_cutted == 3))][mask])
                if mask.any() else pv.PolyData()
            )
            plotter.add_mesh(data.glyph(scale=False, geom=sp, orient=False),
                             opacity=op, color=Colors_An[n_bonds],
                             name=f"An_{n_bonds}")

        data = pv.PolyData(Pos_cutted[Types_cutted == 4])
        plotter.add_mesh(data.glyph(scale=False, geom=pv.Sphere(radius=0.2), orient=False),
                         opacity=0.05, color="gray", name="H")

    pv.global_theme.allow_empty_mesh = True
    plotter = pv.Plotter()
    plotter.add_axes()
    z_min, z_max = np.min(Pos[:, 2]), np.max(Pos[:, 2])
    plotter.add_slider_widget(slide, [z_min, z_max],
                              value=(z_max + z_min) / 2,
                              title="Pos", fmt="%3.3e")
    plotter.show()

Helix.analyze_plot_syst = analyze_plot_syst

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


def plot_structure(self, type_colors={}, type_radius={},
                        sphere=True, parallel_proj=True,
                        draw_bonds=False, periodic_bonds=False,
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
    Pos = self.Pos
    Types = self.Types
    Lims = self.Lims
    periodic=self.periodic


    Pos_cast = self.Pos_cast
    Types_cast = self.Types_cast
    is_cast = len(Pos_cast) > 0


    ##TO ADD CAST PLOTTING

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

        if len(Bonds):
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
