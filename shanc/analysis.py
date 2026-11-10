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
    'H': 1.3, 'He': 2.0, 'Li': 2.0, 'Be': 2.0, 'B': 2.0, 'C': 2.0, 'N': 2.0, 'O': 2.0, 'F': 2.0, 'Ne': 2.0, 'Na': 2.0, 'Mg': 2.0, 'Al': 2.0, 'Si': 2.0, 'P': 2.0, 'S': 2.0, 'Cl': 2.0, 'Ar': 2.0, 'K': 2.0, 'Ca': 2.0, 'Sc': 2.0, 'Ti': 2.0, 'V': 2.0, 'Cr': 2.0, 'Mn': 2.0, 'Fe': 2.0, 'Co': 2.0, 'Ni': 2.0, 'Cu': 2.0, 'Zn': 2.0, 'Ga': 2.0, 'Ge': 2.0, 'As': 2.0, 'Se': 2.0, 'Br': 2.0, 'Kr': 2.0, 'Rb': 2.0, 'Sr': 2.0, 'Y': 2.0, 'Zr': 2.0, 'Nb': 2.0, 'Mo': 2.0, 'Tc': 2.0, 'Ru': 2.0, 'Rh': 2.0, 'Pd': 2.0, 'Ag': 2.0, 'Cd': 2.0, 'In': 2.0, 'Sn': 2.0, 'Sb': 2.0, 'Te': 2.0, 'I': 2.0, 'Xe': 2.0, 'Cs': 2.0, 'Ba': 2.0, 'La': 2.0, 'Ce': 2.0, 'Pr': 2.0, 'Nd': 2.0, 'Pm': 2.0, 'Sm': 2.0, 'Eu': 2.0, 'Gd': 2.0, 'Tb': 2.0, 'Dy': 2.0, 'Ho': 2.0, 'Er': 2.0, 'Tm': 2.0, 'Yb': 2.0, 'Lu': 2.0, 'Hf': 2.0, 'Ta': 2.0, 'W': 2.0, 'Re': 2.0, 'Os': 2.0, 'Ir': 2.0, 'Pt': 2.0, 'Au': 2.0, 'Hg': 2.0, 'Tl': 2.0, 'Pb': 2.0, 'Bi': 2.0, 'Po': 2.0, 'At': 2.0, 'Rn': 2.0, 'Fr': 2.0, 'Ra': 2.0, 'Ac': 2.0, 'Th': 2.0, 'Pa': 2.0, 'U': 2.0, 'Np': 2.0, 'Pu': 2.0, 'Am': 2.0, 'Cm': 2.0, 'Bk': 2.0, 'Cf': 2.0, 'Es': 2.0, 'Fm': 2.0, 'Md': 2.0, 'No': 2.0, 'Lr': 2.0, 'Rf': 2.0, 'Db': 2.0, 'Sg': 2.0, 'Bh': 2.0, 'Hs': 2.0, 'Mt': 2.0, 'Ds': 2.0, 'Rg': 2.0, 'Cn': 2.0, 'Nh': 2.0, 'Fl': 2.0, 'Mc': 2.0, 'Lv': 2.0, 'Ts': 2.0, 'Og': 2.0
}



_FIGSIZE = (12, 8)
_FONT = 20

# ---------------------------------------------------------------------------
# Bond computation
# ---------------------------------------------------------------------------

def _compute_bonds(Pos, Types, type_map, threshold_map, threshold_mult = 1.3):
    """
    Compute the bonds of an oxide system.

    |!| This version must only be used for small systems; it builds the full
    distance matrix and is memory-intensive for large systems.


    Parameters
    ----------
    Pos             : (N, 3) array  – atom positions
    Types           : (N,)   array  – atom type ids
    threshold_type1 : float  – cation–anion bond cutoff (Å), default 2
    threshold_type2 : float  – anion–cation bond cutoff (Å), default 2
    threshold_H     : float  – O–H bond cutoff (Å), default 1.3
    do_count_type_3 : bool   – include type-3 (OH) oxygens in anion count
    cation_type     : int    – type id of the cation, default 1
    anion_type      : int    – type id of the anion,  default 2

    Returns
    -------
    Bonds, Cat_count_An, An_count_Cat, O_count_H, H_count_O
    """

    max_type = np.max(Types)
    Bonds = np.zeros((len(Pos),len(Pos)))

    Thresholds = np.array([[ min(threshold_map[type_map[Types[i_1]]],threshold_map[type_map[Types[i_2]]])  for i_1 in range(len(Pos))] for i_2 in range(len(Pos)) ]) * threshold_mult



    for type_ in range(max_type):
        mask_type = (Types==type_)
        Pos_type = Pos[mask_type]
        Threshold_type = Thresholds[mask_type]

        Dist = sd.cdist(Pos_type,Pos)
        #Remove count with itself
        Dist = Dist + 100 * (Dist <= 1e-8)
        mask_bonds = (Dist < Threshold_type)

        Bonds[mask_type] = Bonds[mask_type] + mask_bonds

    Bonds = (Bonds>=1)

    return Bonds



# Neighbour histogram (scalable, periodic-aware)
def compute_bonds_system(helix,cube=30,rdf_max=5):
    """
    Compute bond counts and RDF distances by slicing the system into sub-cubes.

    Parameters
    ----------
    Pos             : (N, 3) array
    Types           : (N,)   array
    cube            : float  – sub-cube edge length (Å), default 100
    threshold_type1 : float  – cation–anion cutoff (Å), default 2
    threshold_type2 : float  – anion–cation cutoff (Å), default 2
    threshold_H     : float  – O–H cutoff (Å), default 1.3
    periodic        : bool   – apply z-periodic boundary, default True
    Lims            : list   – [[lx,Lx],[ly,Ly],[lz,Lz]], required for PBC
    rdf_max         : float  – RDF cutoff radius (Å), default 5
    cation_type     : int    – type id of the cation, default 1
    anion_type      : int    – type id of the anion,  default 2

    Returns
    -------
    Dist_list, Cat_count_An_tot, An_count_Cat_tot
    """

    Pos = helix.Pos
    Types = helix.Types
    Lims = helix.Lims
    type_map = helix.type_map
    threshold_map = helix.threshold_map
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


    In_trunc = np.array(
        [True] * len(Pos) + [False] * (len(Pos_added) - len(Pos)), dtype=bool
    )


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
                    (Pos_added[:, 0] >= x*cube + lx - rdf_max) & (Pos_added[:, 0] < (x+1)*cube + lx + rdf_max) &
                    (Pos_added[:, 1] >= y*cube + ly - rdf_max) & (Pos_added[:, 1] < (y+1)*cube + ly + rdf_max) &
                    (Pos_added[:, 2] >= z*cube + lz - rdf_max) & (Pos_added[:, 2] < (z+1)*cube + lz + rdf_max)
                )

                Pos_trunc   = Pos_added[mask_p]
                Types_trunc = Types_added[mask_p]


                bonds = _compute_bonds(Pos_trunc, Types_trunc, type_map, threshold_map)

                index_p = Index[mask_p]
                n_index_p = len(index_p)
                index_bonds_p = (index_p.reshape((n_index_p,1,1)) * np.array([[1,0]]) + index_p.reshape((1,n_index_p,1)) * np.array([[0,1]])).reshape((n_index_p*n_index_p,2))
                index_bonds_p = index_bonds_p[bonds.ravel()!=0]

                for index_b in index_bonds_p:
                    Bonds.append(index_b)


    return np.array(Bonds)


def compute_rdf(helix,cube=50,rdf_max=5):
    """
    Compute bond counts and RDF distances by slicing the system into sub-cubes.

    Parameters
    ----------
    Pos             : (N, 3) array
    Types           : (N,)   array
    cube            : float  – sub-cube edge length (Å), default 100
    threshold_type1 : float  – cation–anion cutoff (Å), default 2
    threshold_type2 : float  – anion–cation cutoff (Å), default 2
    threshold_H     : float  – O–H cutoff (Å), default 1.3
    periodic        : bool   – apply z-periodic boundary, default True
    Lims            : list   – [[lx,Lx],[ly,Ly],[lz,Lz]], required for PBC
    rdf_max         : float  – RDF cutoff radius (Å), default 5
    cation_type     : int or sequence of int – cation type ID(s), default 1
    anion_type      : int or sequence of int – anion type ID(s), default 2
    hydroxyl_type   : int or sequence of int – hydroxyl oxygen type ID(s),
                      default 3; use an empty sequence to exclude hydroxyls

    Returns
    -------
    Dist_list, Cat_count_An_tot, An_count_Cat_tot
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
                        # print(np.shape(Distances[T1]))
                        # print(np.shape(Distances[T1][:,T2]))

                        D_rdf = Distances[T1][:,T2].flatten()
                        D_rdf = D_rdf[D_rdf <= rdf_max]
                        Rdf[type_1][type_2] = Rdf[type_1][type_2] + D_rdf.tolist()
    helix.rdf = Rdf
    return Rdf



def _convert_symbol_to_type(type_map,symbol):
    """convert a symbol to multiple types if possible"""
    if type_map == {}: raise ValueError("A symbol was used to select atom types, but no mapping was provided")
    else:
        T = []
        for ty in type_map:
            print(ty,symbol,type_map[ty])
            if type_map[ty] == symbol:
                T.append(ty)
    return T

def _clean_type(helix,ty):
    """clean the types to be more usable"""
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


def plot_rdf(self,type_1,type_2,cube=50, rdf_max=5, density=True,
                        vline=None,
                        title=None,
                        plot=False,
                        file_name="rdf.svg",
                        save=True):
    """
    Plot the RDF between cation (type 1) and anion (type 2) atoms.

    Parameters
    ----------
    Pos, Types      : positions and types array
    rdf_max         : float  – x-axis upper limit (Å)
    periodic        : bool   – use z-PBC
    Lims            : list   – system limits (required for PBC)
    vline           : float or None – reference line position (Å); when None,
                      use the mean nearest cation–anion distance
    density         : bool   – normalise to g(r); False = raw counts
    cation_type     : int    – type id of the cation, default 1
    anion_type      : int    – type id of the anion,  default 2
    title           : str    – plot title (auto-generated if None)
    font_weight     : str    – tick/label font weight
    title_font_weight : str  – title font weight
    """



    rdf = self.rdf
    if not len(rdf):
        rdf = compute_rdf(self,cube=cube,rdf_max=rdf_max)

    T1,label_1 = _clean_type(self,type_1)
    T2,label_2 = _clean_type(self,type_2)

    print(T1,T2)
    print(type_1,type_2)
    print(self.type_map)

    Dist_list = []
    count = 0
    for t1 in T1:
        for t2 in T2:
            Dist_list += rdf[t1-1][t2-1]
    for t2 in T2:
        count += np.sum(self.Types==t2)
    print(T1,T2,count)

    # if vline is None:
    #     vline = _mean_nearest_pair_distance(
    #         Pos, Types, cation_types, anion_types,
    #         periodic=periodic and len(Lims) > 0, Lims=Lims,
    #     )

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

    plot_title = title if title else f"RDF {label_1}–{label_2}"
    # plt.title(plot_title, color=purple)
    plt.xlabel("Distance (Å)", color=purple,size=17)

    # if vline:
    #     tick_locs = [k for k in range(int(rdf_max))] + [vline]
    #     ax.axvline(vline, color=dark_dark_purple, linestyle='--', linewidth=3,
    #            label=f"Bond distance {vline} Å")
    # else:
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




# ---------------------------------------------------------------------------
# Helpers Atom type resolve
# ---------------------------------------------------------------------------

def _require_material(self, expected):
    if self.material != expected:
        raise ValueError(
            f"{expected.capitalize()} analysis requires a {expected} type_map; "
            f"this helix is {self.material!r}."
        )


def _resolve_type_ids(self, selector, defaults, role):
    """Resolve a role selection from symbols or validate explicit type IDs."""
    if selector is None:
        return tuple(int(type_id) for type_id in defaults)

    if isinstance(selector, str):
        requested = selector.casefold()
        selected = tuple(
            int(type_id) for type_id, symbol in self.type_map.items()
            if isinstance(symbol, str) and symbol.casefold() == requested
        )
    else:
        selected = tuple(int(type_id) for type_id in np.atleast_1d(selector))
        missing = [type_id for type_id in selected if type_id not in self.type_map]
        if missing:
            raise ValueError(f"{role} type ID(s) {missing} are not in helix.type_map.")

    if not selected:
        raise ValueError(f"Could not resolve {role} selection {selector!r} from helix.type_map.")
    return selected


def _oxide_analysis_types(self, cation_type=None, anion_type=None, hydrogen_type=None):
    """Resolve oxide role IDs from the helix type map."""
    _require_material(self, "oxide")
    cations = _resolve_type_ids(
        self, cation_type, self.oxide_cation_types, "cation"
    )
    anions = _resolve_type_ids(
        self, anion_type, self.oxide_anion_types, "anion"
    )
    hydrogens = _resolve_type_ids(
        self, hydrogen_type, self.hydrogen_types, "hydrogen"
    ) if hydrogen_type is not None or self.hydrogen_types else ()
    if not cations or not anions:
        raise ValueError(
            "Oxide analysis requires at least one cation type and one anion type "
            "in helix.type_map."
        )
    return cations, anions, hydrogens


def _metal_type_ids(self, type_id=None):
    """Resolve the metallic types to analyse, validating an optional selection."""
    if self.material != "metal":
        raise ValueError(
            f"Metal analysis requires a metal type_map; this helix is {self.material!r}."
        )
    metal_types = tuple(int(type_id) for type_id in self.metal)
    if type_id is not None:
        if isinstance(type_id, str) and not type_id.isdecimal():
            selected = tuple(
                int(mapped_id) for mapped_id, symbol in self.type_map.items()
                if isinstance(symbol, str) and symbol.casefold() == type_id.casefold()
            )
            if not selected or any(mapped_id not in metal_types for mapped_id in selected):
                raise ValueError(f"type_id={type_id!r} is not mapped to a metal element.")
            return selected
        selected_id = int(type_id)
        if selected_id not in metal_types:
            raise ValueError(f"type_id={type_id} is not mapped to a metal element.")
        return (selected_id,)
    return metal_types

def _type_labels(self, type_ids):
    """Return readable element labels for selected type IDs."""
    return ", ".join(
        str(self.type_map.get(type_id, f"type {type_id}"))
        for type_id in type_ids
    )

def _resolve_type_ids(self, selector, defaults, role):
    """
    Resolve a type selector into validated type IDs.

    Parameters
    ----------
    self : Helix
        Helix object containing the type map.
    selector : str, int, sequence, or None
        Element symbol or type ID selection. ``None`` uses ``defaults``.
    defaults : sequence of int
        Type IDs to use when no selector is specified.
    role : str
        Role name used in validation errors.

    Returns
    -------
    tuple of int
        Resolved type IDs.

    Raises
    ------
    ValueError
        If the selection cannot be resolved from the helix type map.
    """
    if selector is None:
        return tuple(int(type_id) for type_id in defaults)

    if isinstance(selector, str):
        requested = selector.casefold()
        selected = tuple(
            int(type_id) for type_id, symbol in self.type_map.items()
            if isinstance(symbol, str) and symbol.casefold() == requested
        )
    else:
        selected = tuple(int(type_id) for type_id in np.atleast_1d(selector))
        missing = [type_id for type_id in selected if type_id not in self.type_map]
        if missing:
            raise ValueError(f"{role} type ID(s) {missing} are not in helix.type_map.")

    if not selected:
        raise ValueError(f"Could not resolve {role} selection {selector!r} from helix.type_map.")
    return selected

def _periodic_images_z(Pos, Lims, margin):
    """
    Create z-periodic images of atoms near the box boundaries.

    Parameters
    ----------
    Pos : array_like, shape (N, 3)
        Cartesian atom coordinates.
    Lims : array_like
        Box limits in ``[[xlo, xhi], [ylo, yhi], [zlo, zhi]]`` form.
    margin : float
        Boundary distance within which atoms are copied.

    Returns
    -------
    tuple
        Coordinates including images and source atom indices. Original atoms
        precede the images.
    """
    lz, Lz = Lims[2]
    Dz = Lz - lz
    top = np.where(Pos[:, 2] > Lz - margin)[0]
    bot = np.where(Pos[:, 2] < lz + margin)[0]
    Pos_all = np.vstack([Pos, Pos[top] - [0, 0, Dz], Pos[bot] + [0, 0, Dz]])
    return Pos_all, np.concatenate([np.arange(len(Pos)), top, bot])

# ---------------------------------------------------------------------------
# RDF plots
# ---------------------------------------------------------------------------

def _style_axes(ax, title, xlabel, ylabel, font_weight, title_font_weight):
    """Apply consistent styling to a matplotlib Axes."""
    for spine in ax.spines.values():
        spine.set_linewidth(5)
    ax.set_title(title, color='black', weight=title_font_weight, fontsize=_FONT)
    ax.set_xlabel(xlabel, color='black', weight=font_weight, fontsize=_FONT)
    ax.set_ylabel(ylabel, color='black', weight=font_weight, fontsize=_FONT)
    ax.tick_params(width=5, length=5, color='black')
    plt.setp(ax.get_yticklabels(), color='black', weight=font_weight, fontsize=_FONT)
    plt.setp(ax.get_xticklabels(), color='black', weight=font_weight, fontsize=_FONT)
    plt.tight_layout()



# ---------------------------------------------------------------------------
# Plot coordination check
# ---------------------------------------------------------------------------

def plot_coordination_metal(self, type_id=None,
                          a_theory=None,
                          threshold_nn=3.2,
                          cube=30,
                          plot=True,
                          font_weight='normal',
                          title_font_weight='normal',
                          show = True):
    """
    Validate a transformed metal structure before MD relaxation by comparing
    nearest-neighbour distances and coordination numbers against ideal FCC
    reference values.

    Parameters
    ----------
    Pos           : (N, 3) array – atom positions (Å)
    Types         : (N,)   array – atom type ids
    type_id       : int    – metal type id, default 1
    a_theory      : float  – FCC lattice parameter (Å); Au default 4.078
    threshold_nn  : float  – neighbour cutoff (Å), default 3.2
    cube          : float  – sub-cube edge (Å), default 30
    plot          : bool   – show matplotlib figures, default True
    font_weight, title_font_weight : str – matplotlib font weight strings

    """

    Pos = self.Pos
    Types = self.Types

    metal_type_ids = _metal_type_ids(self, type_id)
    if a_theory is None:
        symbols = {
            self.type_map[type_id]
            for type_id in metal_type_ids
            if type_id in self.type_map
        }
        fcc_lattice_parameters = {"Au": 4.078, "Ag": 4.086, "Cu": 3.615}
        if len(symbols) != 1 or not symbols <= fcc_lattice_parameters.keys():
            raise ValueError(
                "a_theory must be provided unless one supported FCC metal is selected."
            )
        a_theory = fcc_lattice_parameters[symbols.pop()]

    d_NN     = a_theory / np.sqrt(2)
    d_min_ok = 0.85 * d_NN
    pad      = threshold_nn + 0.2

    metal_pos = Pos[np.isin(Types, metal_type_ids)]
    N = len(metal_pos)
    if N == 0:
        raise ValueError(f"No atoms found for metal type id(s) {metal_type_ids}.")
    print(f"Checking {N} metal atoms (type id(s) {metal_type_ids})")

    lx, ly, lz = np.min(metal_pos, axis=0)
    Lx, Ly, Lz = np.max(metal_pos, axis=0)

    Nx = int((Lx - lx) // cube + 1)
    Ny = int((Ly - ly) // cube + 1)
    Nz = int((Lz - lz) // cube + 1)

    nn_dists    = np.full(N, np.inf)
    cn_arr      = np.zeros(N, dtype=int)
    n_too_close = 0

    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                ind_uniq = (
                    (metal_pos[:, 0] >= x*cube + lx) & (metal_pos[:, 0] < (x+1)*cube + lx) &
                    (metal_pos[:, 1] >= y*cube + ly) & (metal_pos[:, 1] < (y+1)*cube + ly) &
                    (metal_pos[:, 2] >= z*cube + lz) & (metal_pos[:, 2] < (z+1)*cube + lz)
                )
                ind_pad = (
                    (metal_pos[:, 0] >= x*cube + lx - pad) & (metal_pos[:, 0] < (x+1)*cube + lx + pad) &
                    (metal_pos[:, 1] >= y*cube + ly - pad) & (metal_pos[:, 1] < (y+1)*cube + ly + pad) &
                    (metal_pos[:, 2] >= z*cube + lz - pad) & (metal_pos[:, 2] < (z+1)*cube + lz + pad)
                )

                pos_uniq = metal_pos[ind_uniq]
                pos_pad  = metal_pos[ind_pad]
                if len(pos_uniq) == 0:
                    continue

                D = sd.cdist(pos_uniq, pos_pad)
                for local_i, global_i in enumerate(np.where(ind_uniq)[0]):
                    pad_indices = np.where(ind_pad)[0]
                    col = np.where(pad_indices == global_i)[0]
                    if len(col):
                        D[local_i, col[0]] = np.inf

                nn_dists[ind_uniq]  = np.minimum(nn_dists[ind_uniq], np.min(D, axis=1))
                cn_arr[ind_uniq]   += np.sum(D < threshold_nn, axis=1)
                n_too_close        += np.sum(D < d_min_ok)

    bond_strain = (nn_dists - d_NN) / d_NN
    cn_counts   = {cn: int(np.sum(cn_arr == cn)) for cn in range(0, 14)}

    report = {
        'nn_dist_mean': float(np.mean(nn_dists)),
        'nn_dist_std':  float(np.std(nn_dists)),
        'cn_mean':      float(np.mean(cn_arr)),
        'cn_counts':    {k: v for k, v in cn_counts.items() if v > 0},
        'n_too_close':  int(n_too_close),
        'n_isolated':   int(np.sum(cn_arr == 0)),
        'strain_mean':  float(np.mean(bond_strain)),
        'strain_std':   float(np.std(bond_strain)),
    }

    if not plot:
        return report


    cn_vals  = sorted(report['cn_counts'].keys())
    cn_freqs = [report['cn_counts'][c] for c in cn_vals]

    fig, ax = plt.subplots(figsize = _FIGSIZE)

    bars = ax.bar(cn_vals, cn_freqs, color=_PURPLE, edgecolor=_DARK_PURPLE, linewidth=3)

    top_three = sorted(
        [(val / N * 100, bar, val) for bar, val in zip(bars, cn_freqs)],
        key=lambda x: x[0], reverse=True
    )[:3]

    for pct, bar, yval in top_three:
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + max(cn_freqs) * 0.02,
            f"{pct:.1f}%",
            ha='center', va='bottom',
            color=_DARK_PURPLE, weight=font_weight, fontsize=_FONT,
        )

    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.tick_params(width=5, length=5, color=_DARK_PURPLE)
    ax.set_xlim(4, 14)
    ax.set_ylim(0, max(cn_freqs) * 1.15)

    _style_axes(ax, "Coordination number distribution", "Coordination number", "Number of atoms", font_weight, title_font_weight)

    plt.tight_layout()
    plt.savefig("coordination_metal.svg")

    if show:
        plt.show()

    return report


Helix.plot_coordination_metal = plot_coordination_metal



def plot_coordination_oxide(self, cation=None, anion=None, cutoff=2.5,
                            include_H_in_anion=True, show=True,
                            font_weight="normal", title_font_weight="normal"):
    """
    Plot cation and anion coordination-number histograms for an oxide helix.

    The helix's oxide type roles are selected automatically unless ``cation``
    or ``anion`` is provided.

    Parameters
    ----------
    self : Helix
        Helix object containing positions, atom types, and the material type map.
    cation : str, int, or sequence of str or int, optional
        Cation element symbol(s) or type ID(s). Defaults to the cation type IDs
        inferred from ``self.type_map``.
    anion : str, int, or sequence of str or int, optional
        Anion element symbol(s) or type ID(s). Defaults to oxygen type IDs
        inferred from ``self.type_map``.
    cutoff : float, default=2.5
        Maximum cation-anion distance, in angstroms, used to count neighbours.
    include_H_in_anion : bool, default=True
        Include hydrogen neighbours in the anion coordination counts.
        Hydrogen coordination uses a fixed 1.3 angstrom cutoff.
    save : str or None, default="coordination_oxide.svg"
        Output path for the figure. Set to ``None`` to skip saving.
    show : bool, default=True
        Whether to display the figure.
    font_weight : str, default="normal"
        Font weight used for axis labels and tick labels.
    title_font_weight : str, default="normal"
        Font weight used for plot titles.

    Returns
    -------
    dict
        ``cn_cat``, ``cn_an``, and ``an_h`` coordination arrays, cation-anion
        distances in ``dist``, and selected atom indices in ``cat_idx`` and
        ``an_idx``.

    """
    _require_material(self, "oxide")
    cation_types = _resolve_type_ids(
        self, cation, self.oxide_cation_types, "cation"
    )
    anion_types = _resolve_type_ids(
        self, anion, self.oxide_anion_types, "anion"
    )
    hydrogen_types = self.hydrogen_types
    cation_label = _type_labels(self, cation_types)
    anion_label = _type_labels(self, anion_types)
    Pos = np.asarray(self.Pos, dtype=float)
    Types = np.asarray(self.Types)
    rdf_max = 5.0
    cutoff_H = 1.3
    if self.periodic and self.Lims is not None and len(self.Lims) > 0:
        Pos_all, source_indices = _periodic_images_z(
            Pos, self.Lims, max(rdf_max, cutoff_H)
        )
    else:
        Pos_all = Pos
        source_indices = np.arange(len(Pos))
    Types_all = Types[source_indices]

    cat_idx = np.where(np.isin(Types, cation_types))[0]
    an_idx = np.where(np.isin(Types, anion_types))[0]
    cat_all = np.where(np.isin(Types_all, cation_types))[0]
    an_all = np.where(np.isin(Types_all, anion_types))[0]
    hydrogen_all = np.where(np.isin(Types_all, hydrogen_types))[0]

    cat_tree = cKDTree(Pos[cat_idx])
    anion_tree = cKDTree(Pos[an_idx])
    cat_all_tree = cKDTree(Pos_all[cat_all])
    anion_all_tree = cKDTree(Pos_all[an_all])

    cat_neighbours = cat_tree.query_ball_tree(anion_all_tree, rdf_max)
    cat_distances = []
    cn_cat = np.zeros(len(cat_idx), dtype=int)
    for i, neighbour_indices in enumerate(cat_neighbours):
        if neighbour_indices:
            distances = np.linalg.norm(
                Pos_all[an_all[neighbour_indices]] - Pos[cat_idx[i]], axis=1
            )
            cat_distances.extend(distances)
            cn_cat[i] = np.sum(distances < cutoff)

    anion_neighbours = anion_tree.query_ball_tree(cat_all_tree, cutoff)
    cn_an = np.array([len(neighbours) for neighbours in anion_neighbours], dtype=int)

    if len(hydrogen_all):
        hydrogen_tree = cKDTree(Pos_all[hydrogen_all])
        hydrogen_neighbours = anion_tree.query_ball_tree(hydrogen_tree, cutoff_H)
        an_h = np.array(
            [len(neighbours) for neighbours in hydrogen_neighbours], dtype=int
        )
    else:
        an_h = np.zeros(len(an_idx), dtype=int)

    res = {
        "cn_cat": cn_cat,
        "cn_an": cn_an,
        "an_h": an_h,
        "dist": np.asarray(cat_distances),
        "cat_idx": cat_idx,
        "an_idx": an_idx,
    }
    cn_cat = res["cn_cat"]
    cn_an = res["cn_an"] + (res["an_h"] if include_H_in_anion else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    cat_cn_vals = np.arange(10)
    cat_cn_freqs = np.array([np.sum(cn_cat == cn) for cn in cat_cn_vals])

    cat_bars = ax1.bar(cat_cn_vals, cat_cn_freqs, color=_PURPLE,
                       edgecolor=_DARK_PURPLE, linewidth=3)
    for i in np.argsort(cat_cn_freqs)[::-1][:3]:
        if cat_cn_freqs[i]:
            ax1.text(
                cat_bars[i].get_x() + cat_bars[i].get_width() / 2.0,
                cat_cn_freqs[i] + cat_cn_freqs.max() * 0.02,
                f"{100 * cat_cn_freqs[i] / len(cn_cat):.1f}%",
                ha="center", va="bottom", color=_DARK_PURPLE,
                weight=font_weight, fontsize=20,
            )

    ax1.set_xticks(cat_cn_vals)
    ax1.tick_params(width=5, length=5, color="black")
    ax1.set_ylim(0, cat_cn_freqs.max() * 1.15)

    _style_axes(ax1, f"{cation_label} coordination", f"Number of {anion_label} neighbours", f"Number of {cation_label}", font_weight, title_font_weight)

    plt.setp(ax1.get_yticklabels(), color="black", weight=font_weight, fontsize=20)
    plt.setp(ax1.get_xticklabels(), color="black", weight=font_weight, fontsize=20)

    an_cn_vals = np.arange(7)
    an_cn_freqs = np.array([np.sum(cn_an == cn) for cn in an_cn_vals])
    for spine in ax2.spines.values():
        spine.set_linewidth(5)
    an_bars = ax2.bar(an_cn_vals, an_cn_freqs, color=_PURPLE,
                      edgecolor=_DARK_PURPLE, linewidth=3)
    for i in np.argsort(an_cn_freqs)[::-1][:3]:
        if an_cn_freqs[i]:
            ax2.text(
                an_bars[i].get_x() + an_bars[i].get_width() / 2.0,
                an_cn_freqs[i] + an_cn_freqs.max() * 0.02,
                f"{100 * an_cn_freqs[i] / len(cn_an):.1f}%",
                ha="center", va="bottom", color=_DARK_PURPLE,
                weight=font_weight, fontsize=20,
            )
    anion_x_label = f"Number of {cation_label}" + (
        " + H" if include_H_in_anion else ""
    ) + " neighbours"

    ax2.set_xticks(an_cn_vals)
    ax2.set_ylim(0, an_cn_freqs.max() * 1.15)

    _style_axes(ax2, f"{anion_label} coordination", anion_x_label, f"Number of {anion_label} neighbours", font_weight, title_font_weight)

    plt.tight_layout()
    plt.savefig("coordination_oxide.svg")

    if show:
        plt.show()
    return res

Helix.plot_coordination_oxide = plot_coordination_oxide



# ---------------------------------------------------------------------------
# Plot coordination and RDF combined
# ---------------------------------------------------------------------------

def plot_coordination_rdf(self, show=True, coordination_kwargs=None, rdf_kwarg=None):
    """
    Plot coordination-number histograms alongside the RDF.

    Parameters
    ----------
    self : Helix
        Helix object containing positions, atom types, and the material type map.
    show : bool, default=True
        Whether to display the figure.
    coordination_kwargs : dict, optional
        Additional keyword arguments passed to ``plot_coordination_oxide``.
    rdf_kwargs : dict, optional
        Additional keyword arguments passed to ``plot_rdf_cation_anion``.
    """
    if self.material == "oxide":
        plot_coord, plot_rdf = self.plot_coordination_oxide, self.plot_rdf_cation_anion
    elif self.material == "metal":
        plot_coord, plot_rdf = self.plot_coordination_metal, self.plot_rdf_metal
    else:
        raise ValueError(f"Unsupported material type: {self.material}")

    coord = plot_coord(**{"show": show, **(coordination_kwargs or {})})
    rdf   = plot_rdf(**{"show": show, **(rdf_kwarg or {})})
    return {"coordination": coord, "rdf": rdf}

Helix.plot_coordination_rdf = plot_coordination_rdf


# ---------------------------------------------------------------------------
# Oxide analysis helpers (compute + plot)
# ---------------------------------------------------------------------------

def plot_compute_analysis(self, hist_Dens, hist_Si, hist_O,
                     threshold_type1=2, threshold_type2=2, threshold_H=1.3,
                     rdf_max=5, density=True,
                     cation_type=1, anion_type=2):
    """
    Compute RDF histogram and bond-count histograms (used by analyze_mult).

    Returns
    -------
    [hist_Dens, hist_Si, hist_O], Counts
    """

    print("TO REview")

    Pos = self.Pos
    Types = self.Types
    Lims = self.Lims
    periodic = self.periodic

    Dist_list, Cat_count_An, An_count_Cat = compute_rdf(
        Pos, Types,
        threshold_type1=threshold_type1, threshold_type2=threshold_type2,
        threshold_H=threshold_H, periodic=periodic, Lims=Lims, rdf_max=rdf_max,
        cation_type=cation_type, anion_type=anion_type,
    )
    Dist_list = [k for j in Dist_list for k in j]

    N_at       = len(Pos)
    num_inf_15 = np.sum(np.array(Dist_list) < 1.5)
    num_cat_3  = np.sum(Cat_count_An == 3)
    num_cat_5  = np.sum(Cat_count_An == 5)
    num_an_1   = np.sum(An_count_Cat == 1)
    num_an_3   = np.sum(An_count_Cat == 3)
    print(f"Bonds < 1.5 Å  : {num_inf_15}")
    print(f"3-bond cations : {num_cat_3},  5-bond cations : {num_cat_5}")
    print(f"1-bond anions  : {num_an_1},   3-bond anions  : {num_an_3}")

    hist_Dens, radius = np.histogram(Dist_list, hist_Dens[1])
    if density:
        radius    = ((np.roll(radius, 1) + radius) / 2)[1:]
        dr        = radius[1] - radius[0]
        hist_Dens = hist_Dens / (4 * np.pi * radius**2 * dr) / np.sum(Types == cation_type)

    hist_Si = np.histogram(Cat_count_An, hist_Si[1])[0]
    hist_O  = np.histogram(An_count_Cat, hist_O[1])[0]

    Counts = [np.sum(Cat_count_An), np.sum(An_count_Cat)]
    print(Counts)
    return [hist_Dens, hist_Si, hist_O], Counts

Helix.plot_compute_analysis = plot_compute_analysis

def plot_analysis(Counts_Hists, Counts, hist_Dens, hist_Si, hist_O):
    """Update histogram bar heights (used by analyze_mult)."""
    for count, rect in zip(Counts_Hists[0], hist_Dens[2].patches):
        rect.set_height(count)

    for count, rect in zip(Counts_Hists[1], hist_Si[2].patches):
        rect.set_height(count)
    hist_Si[2].set_label(f"Cat–An Bonds {Counts[0]}")

    for count, rect in zip(Counts_Hists[2], hist_O[2].patches):
        rect.set_height(count)
    hist_O[2].set_label(f"An–Cat Bonds {Counts[1]}")


###########THIS FUNCTIONS HAS TO BE REDONE
def analyze_mult(list_Tstep, list_Pos, list_Types,
                 threshold_type1=2, threshold_type2=2, threshold_H=1.3,
                 rdf_max=5, periodic=False, Lims=[],
                 anim=False, save=False, vline=1.609, density=True,
                 cation_type=1, anion_type=2):
    """
    Compute and interactively display RDF and bond-count distributions for
    one or multiple timesteps.

    Parameters
    ----------
    list_Tstep    : list of int – timestep indices
    list_Pos      : list of (N,3) arrays
    list_Types    : list of (N,)  arrays
    rdf_max       : float – RDF cutoff (Å), default 5
    periodic      : bool  – z-periodic boundary
    Lims          : list  – system limits for PBC
    anim          : bool  – animate over timesteps
    save          : bool  – save last frame as analysis.svg
    vline         : float – reference vertical line on RDF plot
    density       : bool  – normalise RDF to g(r)
    cation_type   : int   – type id of the cation, default 1
    anion_type    : int   – type id of the anion,  default 2
    """

    purple           = _PURPLE
    dark_purple      = _DARK_PURPLE
    dark_dark_purple = _DARK_DARK_PURPLE

    fig, ax = plt.subplots()
    if save:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
        plt.rcParams.update({'font.size': 15})
        plt.rcParams['svg.fonttype'] = 'none'
    plt.axis("off")

    if (not save) and (len(list_Tstep) != 1):
        fig.subplots_adjust(bottom=0.25)
        ax_slider = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        slider = Slider(ax=ax_slider, label="Timestep",
                        valmin=list_Tstep[0], valmax=list_Tstep[-1],
                        valinit=list_Tstep[0], valfmt="%d",
                        valstep=list_Tstep)

    num_Cat = np.sum(list_Types[0] == cation_type)
    num_An  = np.sum(list_Types[0] == anion_type)

    plt.subplot(2, 1, 1)
    hist_Dens = plt.hist(np.array([]), bins=100, range=(0, rdf_max),
                         color=purple, edgecolor=dark_purple, linewidth=1, label="A")
    plt.title("RDF Cat–An", color=dark_purple)
    plt.ylabel("g(r)" if density else "Number", color=dark_purple)
    plt.xlabel("Distance (Å)", color=dark_purple)
    plt.xticks([k for k in range(int(rdf_max))] + [vline],
               [k for k in range(int(rdf_max))] + [vline], color=purple)
    plt.axvline(vline, color=dark_purple)
    plt.xlim(0, rdf_max)
    plt.ylim(0, 2 if density else num_An * 1.5)

    plt.subplot(2, 2, 3)
    hist_Si = plt.hist(np.array([]), bins=12, range=(0, 6),
                       color=purple, edgecolor=dark_purple, linewidth=1, label="A")
    plt.title("Cation bond count", color=dark_purple)
    plt.xlabel("Number of bonds", color=dark_purple)
    plt.ylabel(f"Number of type-{cation_type}", color=dark_purple)
    plt.xticks([k + 0.25 for k in range(7)], [k for k in range(7)], color=purple)
    plt.ylim(0, num_Cat * 1.2)

    plt.subplot(2, 2, 4)
    hist_O = plt.hist(np.array([]), bins=12, range=(0, 6),
                      color=purple, edgecolor=dark_purple, linewidth=1, label="A")
    plt.title("Anion bond count", color=dark_purple)
    plt.xlabel("Number of bonds", color=dark_purple)
    plt.ylabel(f"Number of type-{anion_type}", color=dark_purple)
    plt.xticks([k + 0.25 for k in range(7)], [k for k in range(7)], color=purple)
    plt.ylim(0, num_An * 1.2)

    if anim:
        plt.show(block=False)

    list_Counts_Hists, list_Counts = [], []
    for tstep in range(len(list_Tstep)):
        Hist_Counts, Counts = compute_analysis(
            list_Pos[tstep], list_Types[tstep],
            hist_Dens, hist_Si, hist_O,
            threshold_type1=threshold_type1, threshold_type2=threshold_type2,
            threshold_H=threshold_H, periodic=periodic, Lims=Lims,
            rdf_max=rdf_max, density=density,
            cation_type=cation_type, anion_type=anion_type,
        )
        if anim:
            slider.set_val(list_Tstep[tstep])
            plot_analysis(Hist_Counts, Counts, hist_Dens, hist_Si, hist_O)
            plt.pause(0.02)
        list_Counts_Hists.append(Hist_Counts)
        list_Counts.append(Counts)

    def update(val):
        index = list_Tstep.index(val)
        plot_analysis(list_Counts_Hists[index], list_Counts[index],
                      hist_Dens, hist_Si, hist_O)
        plt.draw()

    if not anim and not save and len(list_Tstep) != 1:
        slider.on_changed(update)
    if not anim and not save:
        update(list_Tstep[0])
        plt.show()

    if save:
        update(list_Tstep[-1])
        plt.tight_layout()
        plt.savefig("analysis.svg")


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


def plot_structure(self, point_size=12, type_colors=None,
                        sphere=True, parallel_proj=True,
                        draw_bonds=False, max_bond_dist=2.0,do_fancy=False):
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

    type_colors = {}
    type_radius = {}


    type_map = self.type_map

    if type_map != {}:
        for type in type_map:
            symbol = type_map[type]
            colour = ELEMENT_COLOURS[symbol]
            type_colors[type] = colour
            radius = ELEMENT_RADIUS[symbol]
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
            geom = pv.Sphere(radius=radius)
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

        Bonds = compute_bonds_system(Pos,Types,Lims=Lims,periodic=periodic)

        lines = np.hstack([[2, i_si, i_o] for i_si, i_o in Bonds])
        mesh.lines = lines



        if do_fancy:
            tube = mesh.tube(radius=0.1)
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
