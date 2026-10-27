"""
Helix System Creation for SHAnC

distortion.py contains tools to duplicate lattice slabs, clean oxide surfaces,
transform a slab into a helical geometry, and generate full periodic systems
for simulation. The functions are specialized for oxide/metal structures and
maintain bond and angle connectivity for OH-passivated surfaces.

"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import scipy.spatial.distance as sd

from shanc.read_write import *
from shanc.analysis import *
from shanc.helix_object import *

# distortion.py contains tools to duplicate lattice slabs, clean oxide surfaces,
# transform a slab into a helical geometry, and generate full periodic systems
# for simulation. The functions are specialized for oxide/metal structures and
# maintain bond and angle connectivity for OH-passivated surfaces.

# ---------------------------------------------------------------------------
# Duplication of the lattice
# ---------------------------------------------------------------------------

def duplicate(Nx_list,Ny_list,Nz,Lims,Atom_type,Atom_pos,Bonds_OH=[],Angles_OH=[]):
    """
    duplicate(Nx_list,Ny_list,Nz,Lims,Atom_type,Atom_pos,clean=False,Bonds_OH=[],Angles_OH=[])

    Duplicates a system multiple times.
    Nx_list and Ny_list can be a list or an int.
    Int are used to duplicate a certain amount of time
    Lists are used to create hollow structures. Typically [0,3,17,20] will duplicate for the positions 0,1,2, 17,18,19

    The data OH are used to also duplicate the bonds and angles as the H are added using harmonic potential

    Parameters
    ----------
        Nx_list : list or int,
            See description
        Ny_list : list or int,
            See description
        Nz : int,
            The amount of times the initial structure is duplicated
        Lims : list,
            The limit coordinates of the system
        Atom_type : list,
            The type of the atoms
        Atom_pos : list,
            the position of the atoms
        Bonds_OH : list, optional
            The indices of the bonds OH
        Anlges_OH : list, optional
            The indices of the angles SiOH

    Returns
    -------
        Pos : The updated positions
        Types : The updated Types
        Lims_tot : The updated limits
        Bonds_OH_tot : The updated bonds
        Angles_OH_tot : The updated angles

    """
    lx,Lx = Lims[0]
    ly,Ly = Lims[1]
    lz,Lz = Lims[2]

    Nx_0,Ny_0 = 0,0

    bonds = False
    if len(Bonds_OH) > 0:
        bonds = True

    Pos = []
    Types = []

    # Support either an integer replicate count or a list describing a
    # hollow/partial replication pattern. This makes it possible to generate
    # full slabs, hollow beams, or custom surface geometries.
    flag_x = 0
    flag_y = 0
    if type(Nx_list) is int:
        Nx = Nx_list
        flag_x = 1

    elif len(Nx_list) == 2:
        Nx_0 = Nx_list[0]
        Nx = Nx_list[1]
        flag_x = 1

    else:
        Nx = Nx_list[-1]

    if type(Ny_list) is int:
        Ny = Ny_list
        flag_y = 1

    elif len(Ny_list) == 2:
        Ny_0 = Ny_list[0]
        Ny = Ny_list[1]
        flag_y = 1

    else:
        Ny = Ny_list[-1]

    for x in range(Nx_0, Nx):
        for y in range(Ny_0,Ny):
            # Determine whether the current replicate cell should be included.
            # Full-int replication occurs when a scalar Nx or Ny is used; for
            # list-based hollow structures, include only the specified ranges.
            if flag_x or flag_y or (x >= Nx_list[0] and x < Nx_list[1]) or (x >= Nx_list[2] and x <= Nx_list[3]) or (y >= Ny_list[0] and y < Ny_list[1]) or (y >= Ny_list[2] and y <= Ny_list[3]):
                for z in range(Nz):
                    for at, ty, num in zip(Atom_pos, Atom_type, range(len(Atom_pos))):
                        Pos.append(at + np.array([x*Lx+lx, y*Ly+ly, z*Lz+lz]))
                        Types.append(ty)


    Pos = np.array(Pos).reshape(len(Pos),3)
    min_x,max_x = np.min(Pos[:,0]),np.max(Pos[:,0])
    min_y,max_y = np.min(Pos[:,1]),np.max(Pos[:,1])
    dx, dy  = (max_x), (max_y)
    max_d = max(dx,dy)
    maxx =  max_d * (2**(1/2))
    maxy =  max_d * (2**(1/2))

    Lims_tot = np.array([[-maxx,maxx],[-maxy,maxy],[lz,Lz]])

    num_at = len(Atom_pos)
    Bonds_OH_tot = []
    Angles_OH_tot = []
    if bonds:
        for j in range(Nz):
            for bond in Bonds_OH:
                Bonds_OH_tot.append([bond[0]+num_at*j,bond[1]+num_at*j])
            for angle in Angles_OH:
                Angles_OH_tot.append([angle[0]+num_at*j,angle[1]+num_at*j,angle[2]+num_at*j])
        return Pos,np.array(Types),Lims_tot, Bonds_OH_tot, Angles_OH_tot

    else:
        return Pos,np.array(Types),Lims_tot, [], []


# ---------------------------------------------------------------------------
# Structure Passivation / Cleaning
# ---------------------------------------------------------------------------

def _find_missing_bond_dir(bonded_vecs, sign_x_fallback):
    """
    Return the normalised direction of the most likely missing bond,
    given the vectors from a cation to its currently bonded anions.

    Method: the missing bond points opposite to the centroid of the
    existing bond unit-vectors.  This is exact for symmetric polyhedra
    (tetrahedral Si, octahedral Ti) that have lost exactly one bond.

    Falls back to the Â±x axis when the existing bonds cancel out exactly
    (fully symmetric residual â€” cannot determine a unique direction).

    Parameters
    ----------
    bonded_vecs      : np.ndarray (K, 3)  vectors Cat â†’ bonded O
    sign_x_fallback  : Â±1  used when the heuristic fails

    Returns
    -------
    direction : np.ndarray (3,)  normalised
    """
    if len(bonded_vecs) == 0:
        return np.array([float(sign_x_fallback), 0.0, 0.0])

    norms = np.linalg.norm(bonded_vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)   # avoid div-by-zero
    unit_vecs = bonded_vecs / norms

    missing = -np.sum(unit_vecs, axis=0)
    mag     = np.linalg.norm(missing)

    if mag < 0.1:
        # Bonds cancel --> cannot determine unique missing direction
        return np.array([float(sign_x_fallback), 0.0, 0.0])

    return missing / mag


def _auto_cap_params(Pos, Types, Lims, cation_type, anion_type, cutoff_factor = 1.2, oo_factor = 0.6, cap_missing = 2):

    Pos_Cat = Pos[Types == cation_type]
    Pos_An = Pos[Types == anion_type]
    D = sd.cdist(Pos_Cat, Pos_An)

    z0, Lz_unit = Lims[2][0], Lims[2][1] - Lims[2][0]
    mid_cat = (Pos_Cat[:,2] >= z0 + Lz_unit) & (Pos_Cat[:,2] < z0 + 2*Lz_unit)
    mid_an = (Pos_An[:,2] >= z0 + Lz_unit) & (Pos_An[:,2] < z0 + 2*Lz_unit)

    # 1. Bond threshold from nearest-neighbour distance
    d_bulk = np.median(D[mid_cat].min(axis=1))
    bond_threshold = cutoff_factor * d_bulk

    # 2. Bulk coordination
    counts = np.sum(D < bond_threshold, axis = 1)
    vals, n = np.unique(counts[mid_cat], return_counts=True)
    bulk_coord = int(vals[np.argmax(n)])

    # 3. Bond length from fully coordinated cations
    full = mid_cat & (counts == bulk_coord)
    d_full = D[full]
    bond_length = float(d_full[d_full < bond_threshold].mean())

    # 4. Collision tolerence from O_O nearest-neighbour distance
    D_oo = sd.cdist(Pos_An[mid_an], Pos_An)
    D_oo[D_oo < 1e-8] = np.inf
    collision_tol = oo_factor * float(np.median(D_oo.min(axis=1)))
    cap_max_coord = bulk_coord - cap_missing

    return dict(bond_threshold=bond_threshold, expected_coord=bulk_coord,
                bond_length=bond_length, collision_tol=collision_tol,
                cap_max_coord = cap_max_coord)




def _clean_structure(Pos, Types, Lims, N, periodic=True,
                        cation_type=1, anion_type=2,
                        expected_coord=4,
                        bond_threshold=2.0,
                        bond_length=1.6, h_length=1.0,
                        collision_tol=0.8, min_count = None, cap_max_coord=None, cap_missing=2):
    """metal
    Cleans the surface of an oxide slab by adding OH groups to under-
    coordinated cations, generalised to any cation type.
    """

    auto = _auto_cap_params(Pos, Types, Lims, cation_type, anion_type, cap_missing = cap_missing)
    print("cap missing:", cap_missing)
    expected_coord = auto["expected_coord"] if expected_coord is None else expected_coord
    bond_threshold = auto["bond_threshold"] if bond_threshold is None else bond_threshold
    bond_length    = auto["bond_length"]    if bond_length    is None else bond_length
    collision_tol  = auto["collision_tol"]  if collision_tol  is None else collision_tol
    cap_max_coord  = auto["cap_max_coord"]  if cap_max_coord  is None else cap_max_coord


    oh_length = bond_length + h_length
    Center = [
    (Lims[0][1]*N[0][0] - Lims[0][0]) / 2,
    (Lims[1][1]*N[1][0] - Lims[1][0]) / 2,
    (Lims[2][1]*N[2][0] - Lims[2][0]) / 2,
    ]

    Pos_Cat = Pos[Types == cation_type]
    Pos_An = Pos[Types == anion_type]

    D_all = sd.cdist(Pos_Cat, Pos_An) # (n_cat, n_an)
    Cat_count_An = np.sum(D_all < bond_threshold, axis=1)

    Lz_unit = Lims[2][1] - Lims[2][0]
    z_mid_lo = Lims[2][0] + Lz_unit
    z_mid_hi = Lims[2][0] + 2*Lz_unit

    mid_mask = (Pos_Cat[:,2] >= z_mid_lo) & (Pos_Cat[:,2]< z_mid_hi)
    if min_count is None:
        min_count = int(np.min(Cat_count_An[mid_mask]))
    else:
        min_count = min_count

    if np.min(Cat_count_An) >= expected_coord:
        print("All cations fully coordinated.")
        return Pos, Types, [], []


    # Lack_Cat_mask = (Cat_count_An <= 2)
    Lack_Cat_mask = (Cat_count_An <= cap_max_coord)
    Lack_Cat = Pos_Cat[Lack_Cat_mask]
    Lack_Cat_D_rows = D_all[Lack_Cat_mask]

    print(f"bond_threshold={bond_threshold} angstrom, "
    f"expected_coord={expected_coord}, collision_tol={collision_tol} angstrom")
    print(f" bond count distribution: "
    f"{dict(zip(*np.unique(Cat_count_An, return_counts=True)))}")
    print(f" minimum bond count = {min_count} "
    f"{len(Lack_Cat)} surface cation(s) to cap")

    Atoms_add_pos = []
    Atoms_add_types = []
    Bonds_OH = []
    num_at = len(Pos)
    partner_idx = []
    for Cat, D_row in zip(Lack_Cat, Lack_Cat_D_rows):

        sign_x = np.sign(Cat[0] - Center[0])
        if sign_x == 0:
            sign_x = 1

        # Bonded-O vectors : used by _find_missing_bond_dir if collision fires
        bonded_vecs = Pos_An[D_row < bond_threshold] - Cat # (K, 3)

        # Add OH on the outward face of Cat
        Atom_O_candidate = Cat + sign_x * np.array([bond_length, 0.0, 0.0])

        # We rebuild Pos_An_live each iteration because Types may have changed
        # (earlier iterations convert type-2 O to type-3 Oh via Types[idx]=3).
        Lz_slab = (Lims[2][1] - Lims[2][0])
        Pos_An_live = Pos[(Types == anion_type) | (Types == 3)]
        Pos_An_check = np.vstack([Pos_An_live, Pos_An_live + [0, 0, Lz_slab],
        Pos_An_live + [0, 0, -Lz_slab]])
        if len(Pos_An_live) > 0:
            dist_cand = np.linalg.norm(Pos_An_check - Atom_O_candidate, axis=1)
            collision = float(np.min(dist_cand)) < collision_tol
        else:
            collision = False

        if collision:
            place_dir = _find_missing_bond_dir(bonded_vecs, sign_x)
        else:
            place_dir = np.array([float(sign_x), 0.0, 0.0])

        place_dir[2] = 0.0
        mag = np.linalg.norm(place_dir)
        if mag < 1e-10:
            place_dir = np.array([float(sign_x), 0.0, 0.0])
        else:
            place_dir = place_dir/mag

        Atom_O_candidate2 = Cat + place_dir*bond_length
        dist_cand2 = np.linalg.norm(Pos_An_check-Atom_O_candidate2, axis = 1)

        # if float(np.min(dist_cand2)) < collision_tol:
        #     print("increase bond length or collision_tol")


        Atom_O_add = Cat + place_dir * bond_length
        Atom_H_add = Cat + place_dir * oh_length
        Atoms_add_pos.extend([Atom_O_add, Atom_H_add])
        Atoms_add_types.extend([3, 4])
        Bonds_OH.append([num_at, num_at + 1])

        # Add an H on the opposite surface
        Cat_symm = np.array([2 * Center[0] - Cat[0], Cat[1], Cat[2]])

        Pos_An_trunc = Pos_An[
            (Pos_An[:, 2] > (Cat[2] - 2)) & (Pos_An[:, 2] < (Cat[2] + 2))
        ]
        D_to_symm = np.sum((Pos_An_trunc - Cat_symm.reshape((1, 3))) ** 2, axis=1)
        # Simply take the O closest to the symmetric point of Cat
        Pos_An_cap = Pos_An_trunc[np.argmin(D_to_symm)]

        sign_cap = np.sign(Pos_An_cap[0] - Center[0])
        if sign_cap == 0:
            sign_cap = -sign_x
        Atom_H_cap = Pos_An_cap + sign_cap * np.array([h_length, 0.0, 0.0])
        Atoms_add_pos.append(Atom_H_cap)
        Atoms_add_types.append(4)

        D_global_all = np.sum((Pos - Pos_An_cap) ** 2, axis=1)
        idx_cap = np.argmin(D_global_all)
        Types[idx_cap] = 3
        Bonds_OH.append([idx_cap, num_at + 2])
        num_at += 3
        partner_idx.append(idx_cap)

    if len(Atoms_add_pos) == 0:
        print("No under-coordinated cations found.")
        return Pos, Types, [], []

    Atoms_add_pos = np.array(Atoms_add_pos, dtype=float)
    Atoms_add_types = np.array(Atoms_add_types, dtype=int)

    New_Pos = np.append(Pos, Atoms_add_pos, axis=0)
    New_Types = np.append(Types, Atoms_add_types, axis=0)

    Middle = (
    (New_Pos[:, 2] >= (Lims[2, 1] + Lims[2, 0]))
    & (New_Pos[:, 2] < (Lims[2, 1]*2 + Lims[2, 0]))
    )

    n_caps_total = len(Lack_Cat)
    n_partner_unique = len(set(partner_idx))        # append idx_cap to partner_idx inside the loop
    print("caps:", n_caps_total, "unique partner O:", n_partner_unique)

    mid_types = New_Types
    nTi = np.sum(mid_types == cation_type)
    nO  = np.sum((mid_types == anion_type) | (mid_types == 3))
    nH  = np.sum(mid_types == 4)
    print("Ti:", nTi, "O(2+3):", nO, "H:", nH)
    print("O excess vs 2*Ti:", nO - 2*nTi, " (should equal number of added OH, A)")

    New_Pos = New_Pos[Middle]
    New_Pos = New_Pos - np.array([0, 0, np.min(New_Pos[:, 2])])
    New_Types = New_Types[Middle]

    Bonds_OH_corrected = []
    New_index = (np.cumsum(Middle) - 1) * Middle

    for bond in Bonds_OH:
        if New_index[bond[0]] != 0 and New_index[bond[1]] != 0:
            Bonds_OH_corrected.append(
            [New_index[bond[0]] + 1, New_index[bond[1]] + 1]
            )

    Angles_OH = []
    Pos_Cat_new = New_Pos[New_Types == cation_type]

    for bond in Bonds_OH_corrected:
        O = New_Pos[bond[0] - 1]
        Pos_Cat_trunc = Pos_Cat_new[
        (Pos_Cat_new[:, 2] > (O[2] - 2)) & (Pos_Cat_new[:, 2] < (O[2] + 2))
        ]

        Dist_Cat = np.sum((Pos_Cat_trunc - O.reshape((1, 3)))**2, axis=1)
        Index_Cat_trunc = (
        (New_Types == cation_type)
        & (New_Pos[:, 2] > (O[2] - 2))
        & (New_Pos[:, 2] < (O[2] + 2))
        )
        Index_Cat_bond = np.argmax((np.cumsum(Index_Cat_trunc) - 1) == np.argmin(Dist_Cat)
        )
        Angles_OH.append([Index_Cat_bond + 1, bond[0], bond[1]])

    return New_Pos, New_Types, Bonds_OH_corrected, Angles_OH



# ---------------------------------------------------------------------------
# Geometric Helical transformation
# ---------------------------------------------------------------------------

def geometric_transformation(Pos,Types,Lims, mean = None,D=0,rota=0,enlarge=10, enlarge_z=0, periodic=True,other_mapping=False,params_helix=[]):
    """
    geometric_transformation(Pos,D=0,rota=0,enlarge=10,periodic=True,circling=True,do_rota_transf=False)

    Do the coordinate transformations of the initial cuboid.
    The formula used for the circling is the elliptical grid mapping


    Parameters
    ----------
        Pos : List,
            The transformed position of the atoms

        rota : float, optional
            The amount of turn to do. By default 0
        enlarge : float, optional
            The space in Angstrom that is added in x and y. This is used to create empty space around the helix. By default 10
        periodic : bool, optional
            do the creation of a periodic system after the transformation
        circling : bool, optional
            Circle the basis of the cuboid. This should be used to create helices as is gives much better curvatures. By default True
        do_rota_transf : bool, optional,
            Use the rota transformation instead of the helix one. This should not be used as it gives much worse structure. By default False

    Returns
    -------
        Pos_transfo : The transformed positions
        Lims : the limits of the new system
    """




    mean = np.mean(Pos,axis=0)
    mean[2] = np.min(Pos[:,2])
    Pos = Pos - mean
    print("visualize realign the center")

    x,y,z = Pos.transpose()
    Lx = np.max(x)
    lx = np.min(x)
    Ly = np.max(y)
    ly = np.min(y)

    if len(params_helix):
        LX = params_helix[1]
        LY = params_helix[2]
        Lz = params_helix[0] / 2 / np.pi
    else:
        LX = Lx-lx
        LY = Ly-ly
        Lz = np.max(Pos[:,2]) / 2 / np.pi

    z = z/Lz
    x = (x-lx - LX/2) / LX * 2
    y = (y-ly - LY/2) / LY * 2


    ## Elliptical Mapping
    x_coord = x * (1-1/2*y**2)**(1/2)
    y_coord = y * (1-1/2*x**2)**(1/2)


    ## Rescaling
    x = x_coord * Lx
    y = y_coord * Ly

    Pos_elliptical_map = np.array([x,y,z]).transpose()

    ### The helix transformation ###

    # The equations were determined using a 2d rotation matrix coupled with a rotation of the basis
    # Another way to see the equations is to compute the tangent and two orthogonal normals to the direction of the 1d helix
    # And to transform the cuboid along these directions : one normal corresponds to y, the other to x and the tangent to x

    R = D * rota
    Norm = (Lz**2 + R**2)**(1/2)

    ## STEP 1: Resizing of z and tilting of the base ##
    z_coord = Lz*z + R * x /Norm
    Pos_elliptical_map = np.array([x,y,z_coord]).transpose()

    ## STEP 2: Transformation along y ##
    y_coord = R * np.cos(z*rota) - np.cos(z*rota) * y + Lz * np.sin(z*rota) / Norm * x

    ## STEP 3: Transformation along x ##
    x_coord = R * np.sin(z*rota) - np.sin(z*rota) * y - Lz * np.cos(z*rota) / Norm * x


    ## STEP 4: Add periodic conditions ##
    if periodic:
        z_coord = (z_coord < 0) * (z_coord+Lz*2*np.pi) + (z_coord > Lz*2*np.pi) * (z_coord - Lz * 2 * np.pi) + (z_coord >= 0) * (z_coord <= Lz*2*np.pi) * z_coord

    # Slide to the initial mean
    # Build the transformed coordinates and output bounding box with the
    # requested padding around the helix.
    Pos_transfo = np.array([x_coord, y_coord, z_coord]).transpose()
    Lims_transfo = np.array([
        [np.min(Pos_transfo[:, 0] - enlarge), np.max(Pos_transfo[:, 0]) + enlarge],
        [np.min(Pos_transfo[:, 1] - enlarge), np.max(Pos_transfo[:, 1]) + enlarge],
        [0, np.max(Pos_transfo[:, 2]) + enlarge_z],
    ])

    return Pos_transfo, Lims_transfo



# ---------------------------------------------------------------------------
# System generation
# ---------------------------------------------------------------------------

def create_helix(self,
                file_lattice="", output_name = "helix.data",
                output_name_cast = "helix_cast.data", mass_map = {},  cap_missing=2):

    """
    create_syst(rota,D,pitch,width,thickness,int_thick,_clean_structure=True,periodic=True,circling=True,do_rota_transf=False,file_lattice="beta_quartz.data",do_angles=False)

    This functions creates the whole system using the dimensions stated.
    It creates two files : by default quartz_dupl.data and quartz_int.data which contains the data of the surface and the data of the cast respectively.

    Parameters
    ----------
        rota : float
            the amount of turns made by the helix
        diameter : float
            the diameter of the circle containing the helix.
        pitch : float
            the pitch, or the period of the helix
        width : float
            the width of the helix
        thickness : float
            the thickness of the helix
        _clean_structure : bool, optional
            To clean or not the structure by adding H2O. By default True
        periodic : bool, optional
            do the creation of a periodic system after the transformation
        circling : bool, optional
            Circle the basis of the cuboid. This should be used to create helices as is gives much better curvatures. By default True
        do_rota_transf : bool, optional,
            Use the rota transformation instead of the helix one. This should not be used as it gives much worse structure. By default False
        file_lattice : float, optional,
            the file containing the lattice that will be duplicated
        do_angles : bool, optional
            write the angles SiOH

    Returns
    -------
        Helix : helix
            helix object that contains the information of the system

        Two files are also created : quartz_dupl.data and quartz_int.data by default
    """

    #the asymetric factor for the cast inside, maybe we should look into this
    asym = 1


    #Try to get the values from the object

    rota = self.turns
    diameter = self.diameter
    pitch = self.pitch
    width = self.width
    thickness = self.thickness
    int_thick = self.int_thick
    twist = self.twist
    saturation = self.saturation
    periodic = self.periodic

    if mass_map == {}:
        mass_map = self.mass_map
        if mass_map == {}:
            raise SyntaxError("Missing mass_map, try adding it when defining your helix, or put it as an argument of this function.")
    else:
        self.mass_map = mass_map


    if file_lattice != "":
        self.read_lattice(file_lattice)

    Pos_lattice = self.Pos_lattice
    Types_lattice = self.Types_lattice
    Lims_lattice = self.Lims_lattice
    material = self.material

    if material == "metal":
        saturation = False

    if (self.file_lattice.split("."))[-1] == "xyz":
        print("/!\\ Warning, you are trying to create an helix starting from a xyz file. This is not recommended as they don't contain the information of the cubic cell. A box was infered but could be wrong...")

    Lims_visu = [[np.max(Pos_lattice[:,0]), np.min(Pos_lattice[:,0])], [np.max(Pos_lattice[:,1]), np.min(Pos_lattice[:,1])], [np.max(Pos_lattice[:,2]), np.min(Pos_lattice[:,2])]]
    print("First RDF check on the FCC")

    # visualize_cuboid(Pos_lattice, Types_lattice, Lims_visu)

    # Get the number of duplication needed to get the proper dimensions
    lx = Lims_lattice[0][1] - Lims_lattice[0][0]
    ly = Lims_lattice[1][1] - Lims_lattice[1][0]
    lz = Lims_lattice[2][1] - Lims_lattice[2][0]

    Nx = int(width // lx +1)
    Ny = int(thickness // ly +1)
    Nz = int(pitch // lz + 1)

    Nx_int = int(int_thick // lx +1)
    Ny_int = int(int_thick // ly +1)

    if asym != 1:
        Nx_left = Nx_int
        Ny_left = Ny_int

        Nx_right = round(asym*Nx_int)
        Ny_right = round(asym*Ny_int)

        print(Nx-Nx_right)
        print(Ny-Ny_right)

        Nx_list = [0,Nx_left,Nx-Nx_right,Nx]
        Ny_list = [0,Ny_left,Ny-Ny_right,Ny]
    else:
        print(Nx_int, Ny_int)
        Nx_list = [0,Nx_int,Nx-Nx_int,Nx]
        Ny_list = [0,Ny_int,Ny-Ny_int,Ny]

    N_list = np.array([[Nx],[Ny],[Nz]])

    use_cast = True
    if int_thick == 0:
        use_cast = False
        Nx_list = Nx
        Ny_list = Ny

    print(Nx_list,Ny_list,Nz)


    Pos_lattice[:,2] += lz/4
    Pos_lattice[:, 2] %= lz

    supercell_Lx = Nx*lx
    supercell_Ly = Ny*ly
    supercell_Lz = Nz*lz

    print("Supercell dimensions :", supercell_Lx, supercell_Ly, supercell_Lz)
    print("The unit cell is repeated",Nx, Ny, Nz,"times")

    ## Due to the transformation, the D is not the same as the one experimentally
    # It is however possible to find the D corresponding to the transformation

    # The maximum distance is taken on the ellipse
    W = width
    P = pitch/2/np.pi
    T = thickness


    D_est = (-T+diameter) /2
    N = P / (P*P + D_est*D_est)**(1/2)
    b = abs(2*T*D_est / (W*W*N*N - T*T))

    if b > 1 :
        # The extremum point is the point in the external layer
        D_transfo = D_est
    else:
        # The extremum is in-between
        d0 = (W*W/4 - diameter**2/4) * P**4 * (W*W-T*T)
        d2 = (P*P*diameter**2/4 * (T*T-W*W) + P*P*T*T * (diameter**2/4-W*W/4)) + P**4*W*W
        d4 = (diameter**2/4 *T*T + W*W*P*P)
        delta = (d2**2 - 4*d4*d0)
        if delta > d2**2:
            D_transfo = ((-d2 + delta**(1/2))/2/d4)**(1/2)
        elif delta < 0:
            print("Two possiblities for the D, the higher one has been taken")
            D_transfo = np.real(((-d2 + delta**(1/2))/2/d4)**(1/2))
        else:
            print("Two possiblities for the D, the higher one has been taken")
            D_transfo = ((-d2 - delta**(1/2))/2/d4)**(1/2)
            D_transfo = ((-d2 + delta**(1/2))/2/d4)**(1/2)

    if D_transfo < 0:
        D_transfo = 0

    if twist:
        D_transfo = 0

    ##Surface
    if saturation:
        ## For oxydes --> need to "clean" the structure with OH bonds
        # Create one slab that is corrected then duplicated Nz times
        # The slab contains 3 duplicates as to only take the inside when doing the cleaning
        Pos, Types, Lims_tot, _a, _b = duplicate(Nx,Ny,3,Lims_lattice,Types_lattice,Pos_lattice)
        # visualize_cuboid(Pos, Types)
        Pos, Types, Bonds_OH, _ = _clean_structure(Pos, Types, Lims_lattice, N_list, periodic=True, cap_missing=cap_missing)

        ##Add O and H to the mappings, SHOULD BE MORE GENERAL IN THE FUTURE
        mass_map[3] = 15.9994
        mass_map[4] = 1.0080
        self.mass_map = mass_map

        type_map = self.type_map
        type_map[3] = "O"
        type_map[4] = "H"
        self.type_map = type_map



        # visualize_cuboid(Pos, Types)
        Pos, Types, Lims_tot, Bonds_OH, _ = duplicate(1,1,Nz,Lims_lattice,Types,Pos,Bonds_OH=Bonds_OH)

        # Adapt the pitch to the new z dimension to avoid problems with the periodicity of the system after the transformation
        # ############
        #### This should not be an issue / I don't think we should do that actually!! It might contrain more the system for big helices
        ## I will look into the periodicity problem
        ## It should come from the enlarge_z which was specific to the silica, it should be corrected now
        # ###########

        # pitch_actual = Nz*lz
        # Pos[:,2] *= pitch/pitch_actual
        # Lims_tot[2] *= pitch/pitch_actual

        # visualize_cuboid(Pos, Types, Lims_visu)
        # plot_rdf_type1type2(Pos, Types)

        #Should not be saved by default, maybe add keyword or something
        # write_data("cuboid.data", Pos, Types, Lims=Lims_tot, Bonds_OH=Bonds_OH, mass_map=mass_map)

    else:
        ## Create slab form which we remove the duplicate atoms ##
        Pos, Types, Lims_tot, _a, _b = duplicate(Nx, Ny, Nz,Lims_lattice,Types_lattice,Pos_lattice)

        Bonds_OH, Angles_OH = [], []
        Lims_visu = [[0,Nx*lx], [0, Ny*ly], [0, Nz*lz]]
        write_data("cuboid.data", Pos, Types, Lims=Lims_visu, Bonds_OH=Bonds_OH, mass_map=mass_map)




    ## Add cast to relax against during the dynamic
    if use_cast:
        Angles_OH = []
        print("Separation between helix wall and cast is", int_thick, "Angstrom")
        # Mask to separate the surface and the cast later
        ## Need elliptical mapping for this ##
        mask_int_x = (((Pos[:,0] - Lims_lattice[0][0])//lx >= Nx_list[1]) & ((Pos[:,0] - Lims_lattice[0][0])//lx < Nx_list[2]))
        mask_int_y = (((Pos[:,1] - Lims_lattice[1][0])//ly >= Ny_list[1]) & ((Pos[:,1] - Lims_lattice[1][0])//ly < Ny_list[2]))

        mask_int = mask_int_x & mask_int_y
        mask_surf = ~mask_int

        enlarge_z = (Lims_lattice[2][1] - np.max(Pos_lattice[:,2])) + (np.min(Pos_lattice[:,2] - Lims_lattice[2][0]))
        # Transformation of the cuboid into the helix

        # Mapping of the indices between cast and surface
        Pos_transfo_int = Pos_transfo[mask_int]
        Types_int = Types[mask_int]
        print("Cast after transformation")

        Pos_transfo_surf = Pos_transfo[mask_surf]
        Types_surf = Types[mask_surf]
        print("Surface after transformation")

        Ind_cut = mask_surf
        Ind_sum = np.cumsum(Ind_cut) - 1
        new_bond_surf = []

        # Recovery of OH Bond indices
        for a1,a2 in Bonds_OH:
            i1 = a1 - 1
            i2 = a2 - 1
            if mask_surf[i1] and mask_surf[i2]:
                new_a1 = Ind_sum[i1] + 1
                new_a2 = Ind_sum[i2] + 1
                new_bond_surf.append([new_a1, new_a2])

        # Check if the bonds are not too large after the transformation and if the pbc are respected,
        # if not adapt the position of the atoms
        for a1, a2 in new_bond_surf:
            d = np.linalg.norm(Pos_transfo_surf[a2 - 1] - Pos_transfo_surf[a1 - 1])
            Lz = Lims_tot[2][1] - Lims_tot[2][0]

            delta = Pos_transfo_surf[a2 - 1,2] - Pos_transfo_surf[a1 - 1, 2]
            delta -= Lz*np.round(delta/Lz)
            if d > 10: # Check if bond too large
                dz = Pos_transfo_surf[a2 - 1, 2] - Pos_transfo_surf[a1 - 1, 2]
                print(dz, Lz/2)
                if dz > Lz/2: # Check if pbc respected
                    Pos_transfo_surf[a2 - 1,2] -= Lz
                if dz < -Lz/2:
                    Pos_transfo_surf[a2 - 1,2] += Lz
                print(Pos_transfo_surf[a2 - 1, 2], Lz/2)
                d = np.linalg.norm(Pos_transfo_surf[a1 - 1] - Pos_transfo_surf[a2 - 1])

        # Write data files
        if Pos_transfo_surf.size > 0:
            write_data(output_name,Pos_transfo_surf,Types_surf,Lims_tot,Bonds_OH=new_bond_surf,Angles_OH=Angles_OH, mass_map=mass_map)
        else:
            print("Surface is empty")

        nTi = np.sum(Types == 1)
        nO  = np.sum((Types == 2) | (Types == 3))
        nH  = np.sum(Types== 4)
        print("Ti:", nTi, "O(2+3):", nO, "H:", nH)
        print("O excess vs 2*Ti:", nO - 2*nTi, " (should equal number of added OH, A)")

        write_data(output_name_cast,Pos_transfo_int,Types_int,Lims_tot,Bonds_OH=Bonds_OH,Angles_OH=Angles_OH, mass_map=mass_map)

    ## If we don't add a cast to relax against ##
    else:
        Angles_OH = []
        print("No cast added, the helix is full")

        enlarge_z = (Lims_lattice[2][1] - np.max(Pos_lattice[:,2])) + (np.min(Pos_lattice[:,2] - Lims_lattice[2][0]))
        ## Transformation of the cuboid into the helix ##
        Pos_transfo,Lims_tot = geometric_transformation(Pos, Types, Lims_lattice, D=D_transfo,rota=rota,periodic=periodic,params_helix=[pitch,width,thickness],enlarge_z=enlarge_z)
        Pos_transfo_int = []
        Types_int = []
        Pos_transfo_surf, Types_surf = Pos_transfo, Types

        # visualize_cuboid(Pos_transfo_surf, Types_surf)
        write_data(output_name, Pos_transfo_surf, Types_surf, Lims_tot,
                Bonds_OH=Bonds_OH, Angles_OH=Angles_OH, mass_map=mass_map)


    self.Pos = Pos_transfo_surf
    self.Types = Types_surf
    self.Lims = Lims_tot
    self.Bonds = Bonds_OH
    self.Angles = Angles_OH
    self.Pos_cast = Pos_transfo_int
    self.Types_cast = Types_int
    self.Lims_cast = Lims_tot


helix.create_helix = create_helix
