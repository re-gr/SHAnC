"""
File reading and writting utilities for SHAnC.

This module contains readers, writers, and helpers for XYZ and LAMMPS
structure files, type/mass mapping, and box parsing. It is designed to
support frame-by-frame simulation data extraction and conversion to the
internal arrays used by the SHAnC analysis pipeline.

Functions include:
    read_xyz, write_xyz, read_dump, write_dump, read_data, write_data
    plus auxiliary format parsers for type map and box information.
"""

import numpy as np
import re
from scipy.spatial.distance import cdist
from shanc.helix_object import *
# ---------------------------------------------------------------------------
# Default atomic masses keyed by element symbol
# ---------------------------------------------------------------------------

ELEMENT_MASSES = {
    "H":   1.0080,  "He":  4.0026,
    "Li":  6.941,   "Be":  9.0122,
    "C":  12.011,   "N":  14.007,   "O":  15.9994,
    "F":  18.998,   "Na": 22.990,   "Mg": 24.305,
    "Al": 26.982,   "Si": 28.0855,  "P":  30.974,
    "S":  32.06,    "Cl": 35.45,    "K":  39.098,
    "Ca": 40.078,   "Ti": 47.867,   "Cr": 51.996,
    "Fe": 55.845,   "Ni": 58.693,   "Cu": 63.546,
    "Zn": 65.38,    "Ga": 69.723,   "Ge": 72.63,
    "Zr": 91.224,   "Ag":107.868,   "Sn":118.710,
    "Pt":195.084,   "Au":196.967,   "Pb":207.2,
}

ELEMENT_AT_NUM = {
      1 : "H",       2 : "He",      3 : "Li",      4 : "Be",
      5 : "B",       6 : "C",       7 : "N",       8 : "O" ,
      9 : "F",      10 : "Ne",     11 : "Na",     12 : "Mg",
     13 : "Al",     14 : "Si",     15 : "P",      16 : "S" ,
     17 : "Cl",     18 : "Ar",     19 : "K",      20 : "Ca",
     21 : "Sc",     22 : "Ti",     23 : "V",      24 : "Cr",
     25 : "Mn",     26 : "Fe",     27 : "Co",     28 : "Ni",
     29 : "Cu",     30 : "Zn",     31 : "Ga",     32 : "Ge",
     33 : "As",     34 : "Se",     35 : "Br",     36 : "Kr",
     37 : "Rb",     38 : "Sr",     39 : "Y",      40 : "Zr",
     41 : "Nb",     42 : "Mo",     43 : "Tc",     44 : "Ru",
     45 : "Rh",     46 : "Pd",     47 : "Ag",     48 : "Cd",
     49 : "In",     50 : "Sn",     51 : "Sb",     52 : "Te",
     53 : "I",      54 : "Xe",     55 : "Cs",     56 : "Ba",
     57 : "La",     58 : "Ce",     59 : "Pr",     60 : "Nd",
     61 : "Pm",     62 : "Sm",     63 : "Eu",     64 : "Gd",
     65 : "Tb",     66 : "Dy",     67 : "Ho",     68 : "Er",
     69 : "Tm",     70 : "Yb",     71 : "Lu",     72 : "Hf",
     73 : "Ta",     74 : "W",      75 : "Re",     76 : "Os",
     77 : "Ir",     78 : "Pt",     79 : "Au",     80 : "Hg",
     81 : "Tl",     82 : "Pb",     83 : "Bi",     84 : "Po",
     85 : "At",     86 : "Rn",     87 : "Fr",     88 : "Ra",
     89 : "Ac",     90 : "Th",     91 : "Pa",     92 : "U" ,
     93 : "Np",     94 : "Pu",     95 : "Am",     96 : "Cm",
     97 : "Bk",     98 : "Cf",     99 : "Es",    100 : "Fm",
    101 : "Md",    102 : "No",    103 : "Lr",    104 : "Rf",
    105 : "Db",    106 : "Sg",    107 : "Bh",    108 : "Hs",
    109 : "Mt",    110 : "Ds",    111 : "Rg",    112 : "Cn",
    113 : "Nh",    114 : "Fl",    115 : "Mc",    116 : "Lv",
    117 : "Ts",    118 : "Og"}

def _mass_to_symbol_map(mass_map):
    """
    Build {type_id: element_symbol} from a {type_id: mass} dict by finding
    the nearest match in ELEMENT_MASSES.  Falls back to "X<id>" when no
    element is within 0.5 Da.
    """
    symbol_map = {}
    for tid, mass in mass_map.items():
        best_sym, best_diff = None, float("inf")
        for sym, m in ELEMENT_MASSES.items():
            diff = abs(m - mass)
            if diff < best_diff:
                best_diff, best_sym = diff, sym
        symbol_map[int(tid)] = best_sym if best_diff < 0.5 else "X{}".format(tid)
    return symbol_map


def _symbol_to_mass_map(symbol_map):
    """
    Build the mass map from a symbol map
    """
    mass_map = {}
    for type in symbol_map:
        symbol = symbol_map[type]
        mass = ELEMENT_MASSES[symbol]
        mass_map[type] = mass

    return mass_map


##### ------- XYZ files ------- #####

def read_xyz(file_name, type_map=None, metal=False):
    """
    read_xyz files
    """



    with open(file_name, "r") as f:
        lines = f.readlines()

    list_TSTEP = []
    list_NUM_AT = []
    list_BOX = []
    list_ATOMS = []
    list_TYPE_MAPS = []

    auto_type_map = {} if type_map is None else dict(type_map)
    next_type_id = max(auto_type_map.values(), default=0) + 1

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        num_at  = int(line)
        n_lines_to_skip = num_at   # save BEFORE num_at is modified below
        comment = lines[i + 1].strip() if (i + 1) < len(lines) else ""

        # ---- parse comment line ----
        tstep = 0
        box   = []
        lx = ly = lz = 1.0

        if comment.lower().startswith("timestep"):
            parts = comment.split()
            tstep = int(parts[1])
            lx, ly, lz = float(parts[2]), float(parts[3]), float(parts[4])
            box = [[0.0, lx], [0.0, ly], [0.0, lz]]

        elif "Lattice=" in comment or 'Lattice="' in comment:
            m = re.search(r'[Ll]attice=["\']?([^"\']+)["\']?', comment)
            if m:
                vals = list(map(float, m.group(1).split()))
                lx, ly, lz = vals[0], vals[4], vals[8]
                box = [[0.0, lx], [0.0, ly], [0.0, lz]]
            try:
                m2 = re.search(r'[Tt]ime[Ss]tep=(\d+)', comment)
                if m2:
                    tstep = int(m2.group(1))
            except Exception:
                pass

        # ---- read atoms always store raw coords here ----
        atoms     = []
        positions = []

        #store dict type to atom
        type_map = {}
        for j in range(num_at):
            ls   = lines[i + 2 + j].split()
            elem = ls[0]
            x, y, z = float(ls[1]), float(ls[2]), float(ls[3])

            if elem not in auto_type_map:
                auto_type_map[elem] = next_type_id
                #add mapping type to symbol
                if type(elem) is int:
                    type_map[next_type_id] = ELEMENT_AT_NUM[elem]
                else: type_map[next_type_id] = elem

                next_type_id += 1

            type_at = auto_type_map[elem]

            atoms.append([j + 1, type_at, x, y, z])   # always raw
            positions.append([x, y, z])

        positions = np.array(positions)

        # ---- infer box AFTER reading all atoms ----
        if not box:
            if metal:
                # d_NN = a / sqrt(2) for FCC
                D = sd.cdist(positions, positions)
                np.fill_diagonal(D, np.inf)
                d_NN  = np.min(D)
                a     = d_NN * np.sqrt(2)
                origin = np.min(positions, axis=0)
                positions = positions - origin
                positions = positions % a
                for k in range(len(atoms)):
                    atoms[k][2] = positions[k, 0]
                    atoms[k][3] = positions[k, 1]
                    atoms[k][4] = positions[k, 2]

                box = [[0.0, a],
                       [0.0, a],
                       [0.0, a]]

                # ---- FIX: remove periodic image atoms on the top faces ----
                # The XYZ contains atoms at x=0 AND x=a (same for y, z).
                # When tiling, cell k's top face = cell k+1's bottom face â†’ duplicates.
                # Keeping only the half-open cell [0, a) fixes this at the source.
                tol = 1e-3  # robust against floating point
                atoms_clean = []
                for atom in atoms:
                    x, y, z = atom[2], atom[3], atom[4]
                    on_top_face = (x > a - tol) or (y > a - tol) or (z > a - tol)
                    if not on_top_face:
                        atoms_clean.append(atom)

                n_removed = len(atoms) - len(atoms_clean)
                print(f"Metal half-open cell: removed {n_removed} top-face atoms "
                      f"(a={a:.4f} A…, d_NN={d_NN:.4f} A…)")
                print(f"  Unit cell: {len(atoms_clean)} atoms (was {len(atoms)})")
                atoms = atoms_clean
                num_at = len(atoms_clean)
            else:
                box = []

        list_TSTEP.append(tstep)
        list_NUM_AT.append(num_at)
        list_BOX.append(box)
        list_ATOMS.append(atoms)
        list_TYPE_MAPS.append(type_map)
        i += 2 + n_lines_to_skip

    return list_TSTEP, list_NUM_AT, list_BOX, np.array(list_ATOMS), list_TYPE_MAPS


def write_xyz(file_out, list_TSTEP, list_BOX, list_ATOMS, symbol_map, last_only=False):
    """
    Shared XYZ writer used by both convert_* functions.
    Positions in list_ATOMS must already be Cartesian (not fractional).
    The comment line carries timestep + box info so the file can be
    round-tripped via read_xyz.
    """
    frames = list(zip(list_TSTEP, list_BOX, list_ATOMS))
    if last_only:
        frames = [frames[-1]]

    with open(file_out, "w") as f:
        for tstep, box, atoms in frames:
            atoms = np.asarray(atoms)
            f.write("{}\n".format(len(atoms)))

            if box:
                lx = box[0][1] - box[0][0]
                ly = box[1][1] - box[1][0]
                lz = box[2][1] - box[2][0]
                f.write("timestep {} {:3.6f} {:3.6f} {:3.6f}\n".format(tstep, lx, ly, lz))
            else:
                f.write("timestep {}\n".format(tstep))

            for row in atoms:
                tid = int(row[1])
                sym = symbol_map.get(tid, "X{}".format(tid))
                f.write("{} {:3.6f} {:3.6f} {:3.6f}\n".format(sym, row[2], row[3], row[4]))

    print("Written: {} ({} frame{})".format(file_out, len(frames), "s" if len(frames) > 1 else ""))






##### -------- Dump files -------- #####

def read_dump(dump_file, unscale=False):
    """Read a LAMMPS dump file."""
    flag_step = 0
    flag_num_at = 0
    flag_box_bound = 0
    flag_atoms = 0
    list_TSTEP = []
    list_NUM_AT = []
    list_BOX = []
    list_ATOMS = []
    BOX = []
    list_at_t = []

    for line in open(dump_file, "r"):
        if flag_step:
            list_TSTEP.append(int(line))
            flag_step = 0
        elif flag_num_at:
            list_NUM_AT.append(int(line))
            flag_num_at = 0
        elif flag_box_bound:
            lsplit = line.split()
            BOX.append([float(lsplit[0]), float(lsplit[1])])
            flag_box_bound -= 1
            if not flag_box_bound:
                list_BOX.append(BOX)
        elif flag_atoms:
            if "ITEM: TIMESTEP" in line:
                flag_step = 1
                flag_atoms = 0
                list_ATOMS.append(list_at_t)
                list_at_t = []
            else:
                lsplit = line.split()
                if len(lsplit) != 5:
                    raise TypeError("Expected dump with: id type xs ys zs")
                if unscale:
                    Lx = BOX[0][1] - BOX[0][0]
                    Ly = BOX[1][1] - BOX[1][0]
                    Lz = BOX[2][1] - BOX[2][0]
                    list_at_t.append([int(lsplit[0]), int(lsplit[1]),
                                      float(lsplit[2]) * Lx + BOX[0][0],
                                      float(lsplit[3]) * Ly + BOX[1][0],
                                      float(lsplit[4]) * Lz + BOX[2][0]])
                else:
                    list_at_t.append([int(lsplit[0]), int(lsplit[1]),
                                      float(lsplit[2]), float(lsplit[3]), float(lsplit[4])])
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
    except Exception:
        print("Atoms lost/removed during dynamics — extracting last timestep only.")
        list_TSTEP  = [list_TSTEP[-1]]
        list_NUM_AT = [list_NUM_AT[-1]]
        list_BOX    = [list_BOX[-1]]
        list_ATOMS  = np.array([list_ATOMS[-1]])

    if unscale:
        Lx = BOX[0][1] - BOX[0][0]
        Ly = BOX[1][1] - BOX[1][0]
        Lz = BOX[2][1] - BOX[2][0]
        C_min = np.mean(list_ATOMS[:, :, 2:], axis=1) + np.array([Lx / 2, Ly / 2, Lz / 2])
        C_min[:, 2] = 0
        C_min = C_min.reshape((len(C_min), 1, 3))
        list_ATOMS[:, :, 2:] = list_ATOMS[:, :, 2:] - C_min
        list_ATOMS[:, :, 2] = list_ATOMS[:, :, 2] % Lx
        list_ATOMS[:, :, 3] = list_ATOMS[:, :, 3] % Ly
        list_ATOMS[:, :, 4] = (list_ATOMS[:, :, 4] - BOX[2][0]) % Lz + BOX[2][0]

    return list_TSTEP, list_NUM_AT, list_BOX, np.array(list_ATOMS)


def write_dump(file_name, list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS):
    """Write a LAMMPS dump file."""
    with open(file_name, "w") as file:
        for tstep, num_at, box, atoms in zip(list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS):
            Lx = box[0][1] - box[0][0]
            Ly = box[1][1] - box[1][0]
            Lz = box[2][1] - box[2][0]
            file.write("ITEM: TIMESTEP\n{}\n".format(tstep))
            file.write("ITEM: NUMBER OF ATOMS\n{}\n".format(num_at))
            file.write("ITEM: BOX BOUNDS pp pp pp\n")
            for box_c in box:
                file.write("{:3.6f} {:3.6f}\n".format(box_c[0], box_c[1]))
            file.write("ITEM: ATOMS id type xs ys zs\n")
            for at in atoms:
                file.write("{} {} {:3.6f} {:3.6f} {:3.6f}\n".format(
                    int(at[0]), int(at[1]), at[2], at[3], at[4]))










##### -------- Data files -------- #####

def read_data(file, do_scale=True, atom_style="atom"):
    """
    read_data(file,do_scale=True,atom_style="atom")
    """

    if atom_style == "full":
        BOX = []
        list_at_t = []
        with open(file, "r") as f:
            for line in f:
                lsplit = line.split()
                if not lsplit:
                    continue
                if "xlo" in line or "ylo" in line or "zlo" in line:
                    BOX.append([float(lsplit[0]), float(lsplit[1])])
                if len(lsplit) == 7:
                    Lx = BOX[0][1] - BOX[0][0]
                    Ly = BOX[1][1] - BOX[1][0]
                    Lz = BOX[2][1] - BOX[2][0]
                    if do_scale:
                        list_at_t.append([int(lsplit[0]), int(lsplit[2]),
                                          float(lsplit[4]) / Lx,
                                          float(lsplit[5]) / Ly,
                                          float(lsplit[6]) / Lz])
                    else:
                        list_at_t.append([int(lsplit[0]), int(lsplit[2]),
                                          float(lsplit[4]), float(lsplit[5]), float(lsplit[6])])
                elif len(lsplit) == 6:
                    Lx = BOX[0][1] - BOX[0][0]
                    Ly = BOX[1][1] - BOX[1][0]
                    Lz = BOX[2][1] - BOX[2][0]
                    if do_scale:
                        list_at_t.append([int(lsplit[0]), float(lsplit[1]),
                                          float(lsplit[3]) / Lx,
                                          float(lsplit[4]) / Ly,
                                          float(lsplit[5]) / Lz])
                    else:
                        list_at_t.append([int(lsplit[0]), float(lsplit[1]),
                                          float(lsplit[3]), float(lsplit[4]), float(lsplit[5])])
        return [BOX], np.array([list_at_t])

    elif atom_style == "atom":
        Lims = []
        Atom_types = []
        Atom_pos = []
        mass_map = {}          # {type_id: mass}
        in_masses   = False
        in_atoms    = False

        for line in open(file):
            lsplit = line.split()
            if not lsplit:
                continue

            # --- section headers ---
            if lsplit[0] == "Masses":
                in_masses = True
                in_atoms  = False
                continue
            if lsplit[0] == "Atoms":
                in_masses = False
                in_atoms  = True
                continue
            # any other keyword header ends Masses section
            if in_masses and len(lsplit) >= 1 and lsplit[0].isalpha():
                in_masses = False

            # --- box bounds ---
            if len(lsplit) >= 4 and lsplit[2] in ("xlo", "ylo", "zlo"):
                try:
                    Lims.append([float(lsplit[0]), float(lsplit[1])])
                except ValueError:
                    pass
                continue

            # --- masses section: "type_id  mass  [# comment]" ---
            if in_masses:
                print(lsplit[1])
                try:
                    mass_map[int(lsplit[0])] = float(lsplit[1])
                except (ValueError, IndexError):
                    pass
                continue

            # --- atoms section ---
            if in_atoms:
                # Supported formats:
                #   atom_style atom  : id  type  x  y  z
                #   atom_style charge: id  type  charge  x  y  z
                #   lammps full      : id  mol  type  charge  x  y  z  (7 cols)
                try:
                    if len(lsplit) == 5:
                        # id type x y z
                        Atom_types.append(int(lsplit[1]))
                        Atom_pos.append([float(lsplit[2]), float(lsplit[3]), float(lsplit[4])])
                    elif len(lsplit) == 6:
                        # id type charge x y z
                        Atom_types.append(int(lsplit[1]))
                        Atom_pos.append([float(lsplit[3]), float(lsplit[4]), float(lsplit[5])])
                    elif len(lsplit) == 7:
                        # id mol type charge x y z
                        Atom_types.append(int(lsplit[2]))
                        Atom_pos.append([float(lsplit[4]), float(lsplit[5]), float(lsplit[6])])
                except (ValueError, IndexError):
                    pass
                continue

        Atom_pos = np.array(Atom_pos, dtype=float)

        # --- infer box from positions if not found in file ---
        if len(Lims) < 3:
            print("Warning: box bounds not found in '{}', inferring from atom positions.".format(file))
            # Use a padding equal to half the smallest inter-atom gap (or 0.5 Å minimum)
            if len(Atom_pos) > 1:
                dists = cdist(Atom_pos, Atom_pos)
                np.fill_diagonal(dists, np.inf)
                padding = max(np.min(dists) / 2.0, 0.5)
            else:
                padding = 0.5
            Lims = [
                [np.min(Atom_pos[:, 0]) - padding, np.max(Atom_pos[:, 0]) + padding],
                [np.min(Atom_pos[:, 1]) - padding, np.max(Atom_pos[:, 1]) + padding],
                [np.min(Atom_pos[:, 2]) - padding, np.max(Atom_pos[:, 2]) + padding],
            ]

        # --- normalise z to start at 0 ---
        z_min = np.min(Atom_pos[:, 2])
        Atom_pos = Atom_pos - np.array([0.0, 0.0, z_min])
        Lims[2][1] -= Lims[2][0]
        Lims[2][0]  = 0.0

        return np.array(Lims), np.array(Atom_types), Atom_pos, mass_map


def write_data(file_name, Pos, Types, Lims,
               test_particle=False,
               Bonds_OH=[], Angles_OH=[],
               mass_map=None,
               # Legacy parameter kept for backwards compatibility
               Types_masses=None, input_file = False):
    """
    Write a LAMMPS data file.

    mass_map : dict  {type_id (int): mass (float)}
               Takes priority over Types_masses.
               Falls back to well-known defaults for common elements
               (Si=28.0855, O=15.9994, H=1.0080, Au=196.967, …).
               Unknown types get mass = 1.0 with a warning.

    Types_masses : list of strings "type_id mass"  (legacy, still accepted)
    """

    # ---- Build the effective mass lookup ----
    # Priority: mass_map arg > Types_masses legacy list > built-in defaults
    _DEFAULT_TYPE_MASSES = {
        1: 28.0855,   # Si (quartz convention)
        2: 15.9994,   # O
        3: 15.9994,   # Oh
        4: 1.0080,    # H
        5: 28.0855,   # Si
        6: 15.9994,   # Oh
        7: 1.0080,    # H
    }
    effective_mass = dict(_DEFAULT_TYPE_MASSES)

    if Types_masses is not None:
        for entry in Types_masses:
            tokens = entry.split()
            try:
                effective_mass[int(tokens[0])] = float(tokens[1])
            except (ValueError, IndexError):
                pass


    if mass_map is not None:
        effective_mass.update(mass_map)

    # ---- Derived flags ----
    unique_types = np.unique(Types)
    H_present = any(t in Types for t in (3, 4, 7))
    bonds  = len(Bonds_OH)  > 0
    angles = len(Angles_OH) > 0

    with open(file_name, "w") as file:
        file.write("\n")
        file.write("{} atoms\n".format(len(Types)))

        if bonds and H_present:
            file.write("{} bonds\n".format(len(Bonds_OH)))
        if angles and H_present:
            file.write("{} angles\n".format(len(Angles_OH)))
        file.write("\n")

        if test_particle:
            file.write("{} atom types\n".format(int(np.max(Types) - np.min(Types) + 1)))
        else:
            file.write("{} atom types\n".format(int(np.max(Types))))

        if bonds and H_present:
            file.write("1 bond types\n")
        if angles and H_present:
            file.write("1 angle types\n")

        # Box
        file.write("\n")
        lo, hi = Lims[0]
        file.write("{:3.6f} {:3.6f} xlo xhi\n".format(lo, hi))
        lo, hi = Lims[1]
        file.write("{:3.6f} {:3.6f} ylo yhi\n".format(lo, hi))
        lo, hi = Lims[2]
        file.write("{:3.6f} {:3.6f} zlo zhi\n".format(lo, hi))
        file.write("\n")

        # Masses — write only for types that actually appear
        file.write("Masses\n\n")
        for t in sorted(unique_types):
            t = int(t)
            if t in effective_mass:
                m = effective_mass[t]
            else:
                print("Warning: no mass for type {}; using 1.0".format(t))
                m = 1.0
            file.write("{} {:3.6f}\n".format(t, m))
        file.write("\n")

        # Atoms
        file.write("Atoms\n\n")
        for num, pos, typ in zip(range(len(Pos)), Pos, Types):
            idx = str(num + 1)
            if input_file:
                file.write("{} {} 0.0 {:3.6f} {:3.6f} {:3.6f}\n".format(
                    idx, int(typ), pos[0], pos[1], pos[2]))
            else:
                file.write("{} 1 {} 0.0 {:3.6f} {:3.6f} {:3.6f}\n".format(
                    idx, int(typ), pos[0], pos[1], pos[2]))

        if bonds and H_present:
            file.write("\nBonds\n\n")
            for num, bond in enumerate(Bonds_OH):
                file.write("{} 1 {} {}\n".format(num + 1, bond[0], bond[1]))

        if angles and H_present:
            file.write("\nAngles\n\n")
            for num, angle in enumerate(Angles_OH):
                file.write("{} 1 {} {} {}\n".format(num + 1, angle[0], angle[1], angle[2]))

        print("Written:", file_name)






###### ------ Conversion of files ------ #######

### Data to xyz ###
def convert_data_to_xyz(file, last_only=False):
    """
    Convert a LAMMPS data file to XYZ format.
    Element symbols are resolved automatically from the Masses section via
    nearest-mass lookup in ELEMENT_MASSES.
    """
    Lims, Atom_types, Atom_pos, mass_map = read_data(file, do_scale=False, atom_style="atom")
    print(mass_map)
    # print(len(Atom_types))
    # print(Atom_pos.min(0), Atom_pos.max(0))
    # print(((Atom_pos < Lims[:,0]) | (Atom_pos > Lims[:,1])).any(axis=1).sum())
    # L = Lims[:,1] - Lims[:,0]
    # Atom_pos = Lims[:,0] + np.mod(Atom_pos - Lims[:,0], L)
    symbol_map = _mass_to_symbol_map(mass_map)

    atoms = np.column_stack([
        np.arange(1, len(Atom_types) + 1),  # id
        Atom_types,                           # type
        Atom_pos,                             # x y z
    ])

    file_out = file.rsplit(".", 1)[0] + ".xyz"
    write_xyz(file_out,
               list_TSTEP=[0],
               list_BOX=[Lims.tolist()],
               list_ATOMS=[atoms],
               symbol_map=symbol_map)
    return Atom_pos, Atom_types, Lims

### Dump to xyz ###
def convert_dump_to_xyz(file, last_only=False, data_file=None, type_map=None):
    """
    Convert a LAMMPS dump file to XYZ format.

    Symbol resolution priority:
      1. type_map  – explicit {type_id: symbol}
      2. data_file – companion .data file, labels inferred from Masses section
      3. Fallback  – "X<type_id>"

    Coordinates are unscaled to Cartesian automatically via read_dump(unscale=True).
    """
    print("Reading dump file:", file)
    list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS = read_dump(file, unscale=True)

    symbol_map = {}
    if data_file is not None:
        _, _, _, mass_map = read_data(data_file, do_scale=False, atom_style="atom")
        symbol_map = _mass_to_symbol_map(mass_map)
    if type_map is not None:
        symbol_map.update({int(k): v for k,v in type_map.items()})

    file_out = file.rsplit(".", 1)[0] + ".xyz"
    write_xyz(file_out,
               list_TSTEP=list_TSTEP,
               list_BOX=list_BOX,
               list_ATOMS=list_ATOMS,
               symbol_map=symbol_map,
               last_only=last_only)
    return list_TSTEP, list_BOX, list_ATOMS


### xyz to data input file for create_sys ###
def _infer_box_from_images(Pos, tol_image=0.05, tol_neg=1e-3):
    """
    Infer exact (Lx, Ly, Lz) from periodic-image atoms in a raw XYZ.

    Atoms with coord < 0 on axis k are left-side images:
        coord_image = coord_true - L  =>  L = coord_inside - coord_image

    For each (outside, inside) candidate pair on axis k, the candidate L
    is accepted only when shifting the outside atom by L brings it within
    tol_image Angstroms of a real inside atom in full 3D.  This rejects
    spurious candidates (bond distances, sub-periods) reliably.

    Falls back to coordinate range for axes with no negative atoms.

    Parameters
    ----------
    Pos        : np.ndarray (N, 3)
    tol_image  : float  3D residual tolerance for accepting a candidate L
    tol_neg    : float  threshold below 0 to consider an atom "outside"

    Returns
    -------
    Ls : list of 3 floats  [Lx, Ly, Lz]
    """
    Pos = np.asarray(Pos, dtype=float)
    Ls  = [None, None, None]

    for axis in range(3):
        coords       = Pos[:, axis]
        outside_mask = coords < -tol_neg
        inside_mask  = ~outside_mask

        if not np.any(outside_mask):
            continue

        outside_atoms = Pos[outside_mask]
        inside_atoms  = Pos[inside_mask]
        accepted      = []

        for out_at in outside_atoms:
            for in_coord in inside_atoms[:, axis]:
                L_cand = in_coord - out_at[axis]
                if L_cand <= tol_neg:
                    continue
                # shift the outside atom by L_cand along this axis
                target       = out_at.copy()
                target[axis] += L_cand
                # check 3D residual against all inside atoms
                dists    = np.linalg.norm(inside_atoms - target, axis=1)
                min_dist = np.min(dists)
                if min_dist < tol_image:
                    accepted.append(L_cand)

        if not accepted:
            continue

        # take the modal cluster (most frequent value within 1e-3)
        accepted  = np.array(sorted(accepted))
        tol_clust = 1e-3
        best_L, best_n = accepted[-1], 0
        i = 0
        while i < len(accepted):
            mask = np.abs(accepted - accepted[i]) < tol_clust
            n    = int(mask.sum())
            if n > best_n or (n == best_n and accepted[i] > best_L):
                best_n = n
                best_L = float(np.mean(accepted[mask]))
            i += max(n, 1)

        Ls[axis] = best_L
        print(f"  [infer_box] axis {'xyz'[axis]}: "
              f"L = {best_L:.6f} Å  ({int(np.any(outside_mask))} image atom(s) used)")

    # fallback for axes with no negative atoms
    for axis in range(3):
        if Ls[axis] is None:
            c      = Pos[:, axis]
            L_fall = float(np.ptp(c))
            Ls[axis] = L_fall
            print(f"  [infer_box] axis {'xyz'[axis]}: "
                  f"no negative atoms, fallback L = {L_fall:.6f} Å")

    return Ls


def wrap_and_deduplicate(Atom_pos, Lims, tol=1e-3):
    """
    Fold every atom into the half-open unit cell [0, L) on each axis,
    then remove atoms that are duplicates within `tol` Angstroms.

    This replicates ASE's atoms.wrap() behaviour for orthogonal boxes.

    Parameters
    ----------
    Atom_pos : np.ndarray (N, 3)  — Cartesian positions
    Lims     : array-like (3, 2)  — [[xlo,xhi],[ylo,yhi],[zlo,zhi]]
    tol      : float              — duplicate-detection threshold (Å)

    Returns
    -------
    pos_wrapped : np.ndarray (M, 3)
    keep_idx    : np.ndarray (M,)  — original indices of kept atoms
    """
    Atom_pos = np.asarray(Atom_pos, dtype=float)
    Lims     = np.asarray(Lims,     dtype=float)

    origin = Lims[:, 0]                        # (xlo, ylo, zlo)
    L      = Lims[:, 1] - Lims[:, 0]          # (Lx,  Ly,  Lz)

    # step 1: fold into [0, L)
    frac        = (Atom_pos - origin) / L      # fractional coords
    frac_folded = frac % 1.0                   # wrap to [0, 1)
    pos_wrapped = frac_folded * L              # back to Cartesian in [0, L)

    for axis in range(3):
        near_top = np.abs(pos_wrapped[:,axis] - L[axis]) < tol
        pos_wrapped[near_top, axis] = 0.0

    # step 2: remove duplicates that landed on the same site
    # Two atoms are duplicates when their folded positions agree within tol
    # on every axis. We keep the one with the lower original index.
    grid = np.round(pos_wrapped / tol).astype(int)
    _, unique_idx = np.unique(grid, axis=0, return_index=True)
    unique_idx    = np.sort(unique_idx)        # preserve original ordering

    n_removed = len(Atom_pos) - len(unique_idx)
    if n_removed:
        print(f"[wrap_and_deduplicate] {n_removed} duplicate(s) removed "
              f"({len(Atom_pos)} → {len(unique_idx)} atoms)")
    else:
        print(f"[wrap_and_deduplicate] No duplicates found "
              f"({len(unique_idx)} atoms)")

    return pos_wrapped[unique_idx], unique_idx


def validate_unit_cell(Atom_pos, Atom_types, Lims, expected_dist=1.613, tol=0.1, type1=1, o_type=2, o_neighbours = 4):
    """Check that every type 1 has exactly o_neighbours O neighbours at expected_dist±tol, using PBC."""
    Lx = Lims[0][1] - Lims[0][0]
    Ly = Lims[1][1] - Lims[1][0]
    Lz = Lims[2][1] - Lims[2][0]
    L  = np.array([Lx, Ly, Lz])

    Pos_type1 = Atom_pos[Atom_types == type1]
    Pos_o  = Atom_pos[Atom_types == o_type]

    all_ok = True
    for i, t1 in enumerate(Pos_type1):
        bonds = []
        for o in Pos_o:
            delta = o - t1
            delta -= L * np.round(delta / L)   # minimum image
            d = np.linalg.norm(delta)
            if abs(d - expected_dist) < tol:
                bonds.append(round(d, 4))
        if len(bonds) != o_neighbours:
            print(f"  [!] type 1 {i}: {len(bonds)} bond(s) found at {bonds} "
                  f"— expected {o_neighbours} x {expected_dist} angstrom")
            all_ok = False
    if all_ok:
        print(f"  Unit cell OK: all Si have 4 O neighbours around {expected_dist} Å")
    return all_ok




# def convert_xyz_to_data(file, type_map=None, mass_map=None, metal=False,
#                          Lx=None, Ly=None, Lz=None, use_box = False):

#     list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS = read_xyz(
#         file, type_map=type_map, metal=metal
#     )
#     Pos   = list_ATOMS[-1][:, 2:]
#     Types = list_ATOMS[-1][:, 1].astype(int)

#     box = list_BOX[-1]
#     if use_box and len(box)==3:
#         Lims = np.array(box)
#     elif Lx is not None and Ly is not None and Lz is not None:   # explicit
#         print("[convert_xyz_to_data] Using user-supplied box dimensions.")
#         Lims = np.array([[0.0, Lx], [0.0, Ly], [0.0, Lz]])

#     else:                                           # try to infer
#         print("[convert_xyz_to_data] No box in file and none supplied — "
#               "inferring from coordinates (see warning below).")
#         Ls =  _infer_box_from_images(Pos)
#         Lims = np.array([[0.0, Ls[0]], [0.0, Ls[1]], [0.0, Ls[2]]])

#     Pos, keep_idx = wrap_and_deduplicate(Pos, Lims)
#     Types = Types[keep_idx]
#     validate_unit_cell(Pos, Types, Lims, expected_dist=1.96, tol = 0.2)

#     if mass_map is None and type_map is not None:
#         mass_map = {tid: ELEMENT_MASSES[elem]
#                     for elem, tid in type_map.items()
#                     if elem in ELEMENT_MASSES}

#     file_out = file.rsplit(".", 1)[0] + ".data"
#     write_data(file_out, Pos, Types, Lims, mass_map=mass_map)
#     return Pos, Types, Lims

def convert_xyz_to_data(file, type_map=None, mass_map=None, metal=False,
                        Lx=None, Ly=None, Lz=None, use_box=False,
                        edge_shift=(0.0, 0.0, 0.0)):

    list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS = read_xyz(
        file, type_map=type_map, metal=metal
    )
    Pos   = list_ATOMS[-1][:, 2:]
    Types = list_ATOMS[-1][:, 1].astype(int)

    box = list_BOX[-1]
    if use_box and len(box) == 3:
        Lims = np.array(box)
    elif Lx is not None and Ly is not None and Lz is not None:
        print("[convert_xyz_to_data] Using user-supplied box dimensions.")
        Lims = np.array([[0.0, Lx], [0.0, Ly], [0.0, Lz]])
    else:
        print("[convert_xyz_to_data] No box in file and none supplied, "
              "inferring from coordinates (see warning below).")
        Ls = _infer_box_from_images(Pos)
        Lims = np.array([[0.0, Ls[0]], [0.0, Ls[1]], [0.0, Ls[2]]])

    # Move atoms off the cell faces so cell indexing is unambiguous
    shift = np.asarray(edge_shift, dtype=float)
    if np.any(shift != 0.0):
        Pos = Pos + shift
        print(f"[convert_xyz_to_data] Shifted atoms by {shift} A before wrapping.")

    Pos, keep_idx = wrap_and_deduplicate(Pos, Lims)
    Types = Types[keep_idx]
    validate_unit_cell(Pos, Types, Lims, expected_dist=1.96, tol=0.2)

    if mass_map is None and type_map is not None:
        mass_map = {tid: ELEMENT_MASSES[elem]
                    for elem, tid in type_map.items()
                    if elem in ELEMENT_MASSES}

    # Report anything still close to a face (x or y)
    for axis, name in ((0, "x"), (1, "y")):
        L = Lims[axis][1] - Lims[axis][0]
        f = (Pos[:, axis] - Lims[axis][0]) / L
        dist = np.minimum(f, 1 - f) * L
        n_close = int(np.sum(dist < 1e-3))
        if n_close:
            print(f"[convert_xyz_to_data] WARNING: {n_close} atoms within 1e-3 A of a {name} face")

    file_out = file.rsplit(".", 1)[0] + ".data"
    write_data(file_out, Pos, Types, Lims, mass_map=mass_map)
    return Pos, Types, Lims



def read_structure_from_file(file, do_scale=True, atom_style="full"):
    """
    read_structure_from_data(file, do_scale=True, atom_style="full")

    This function tries to read a structure file from a file.
    The format of the file should be .data, .xyz or a dump file

    """
    format = file.split(".")[-1]
    if format == "data":
        Lims, Atom_types, Atom_pos, mass_map = read_data(file, do_scale=do_scale, atom_style=atom_style)
        type_map = _mass_to_symbol_map(mass_map)

    elif format == "xyz":
        list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS, list_TYPE_MAP = read_xyz(file, type_map=None, metal=False)
        Lims = list_BOX[-1]
        Atoms_last_step = list_ATOMS[-1]
        Atom_types = Atoms_last_step[:,1]
        Atom_pos = Atoms_last_step[:,2:]
        type_map = list_TYPE_MAP[-1]
        mass_map = _symbol_to_mass_map(type_map)

    else:
        list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS = read_dump(file, unscale=False)
        Lims = list_BOX[-1]
        Atoms_last_step = list_ATOMS[-1]
        Atom_types = Atoms_last_step[:,1]
        Atom_pos = Atoms_last_step[:,2:]
        print("System extracted from a dump file, no symbol mapping are done. If one wants it to be done, one should use....")
        mass_map = {}
        type_map = {}

    return Atom_pos, Atom_types, Lims, mass_map, type_map


def create_helix_from_file(file, do_scale=True, atom_style="full"):
    """
    read_structure_from_data(file, do_scale=True, atom_style="full")

    This function tries to read a structure file from a file.
    The format of the file should be .data, .xyz or .lmmpstrj

    SHould be used to display after dynamics

    """


    return
