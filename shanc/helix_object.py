"""
Helix object definition for SHAnC
"""


from shanc.read_write import *


_METAL_ELEMENTS = frozenset(
    "Li Be Na Mg Al K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi "
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv".split()
)


class Helix:
    """
    Helix object
    """

    def __init__(self, pitch = 0., width = 0., thickness = 0., diameter = 0., turns = 1, file_lattice = ""):
        """
        Helix(pitch = 0., width = 0., thickness = 0., diameter = 0., turns = 1, file_lattice = "")

        Constructs the helix object containing the atoms and the casts.
        If one provides the "file_lattice", the lattice will be directly loaded.


        Attributes
        ----------
            pitch : float
                the pitch, or the period of the helix
            width : float
                the width of the helix
            thickness : float
                the thickness of the helix
            diameter : float
                the diameter of the circle containing the helix.
            turns : float
                the amount of turns made by the helix
            Pos : list
                The position of the atoms
            Types : list
                The types of the atoms
            Lims : list
                The limit coordinates of the system
            Bonds : list
                The list of the bonds typically used for analysis
            Angles : list
                The angles of the OH bonds used to complete the system
            type_map : dict
                the mapping of the type of atom used by LAMMPS and the atomic name
            mass_map : dict
                the mapping of the type of atom used by LAMMPS and the atomic mass
            cast : bool
                If a cast is present or not.
            Pos_cast : list
                The position of the atoms of the cast
            Types_cast : list
                The type of the atoms of the cast
            Lims_cast : list
                The limit coordinates of the cast
            Pos_lattice : list
                The position of the atoms for the lattice
            Types_lattice : list
                The types of the atoms for the lattice
            Lims_lattice : list
                The limit coordinates for the lattice
            twist : bool
                If the helix is a twisted ribbon or not
            saturation : bool
                If the helix has to be saturated or not
            periodic : bool
                If the system should be treated as periodic or not
            file_lattice : str
                The name of the file containing the lattice.

        Parameters
        ----------
            None
                This is just an initialisation

        Returns
        -------
            helix : helix
                The Helix object.
        """

        self.pitch = pitch
        self.width = width
        self.thickness = thickness
        self.diameter = diameter
        self.turns = turns


        self.Pos = []
        self.Types = []
        self.Lims = []
        self.Bonds = []
        self.Angles = []
        self.type_map = {}
        self.mass_map = {}

        #Values cast
        self.cast = False
        self.int_thick = 0.
        self.Pos_cast = []
        self.Types_cast = []
        self.Lims_cast = []

        #Values lattice
        self.Pos_lattice = []
        self.Types_lattice = []
        self.Lims_lattice = []

        #Values Twist
        self.twist = False

        #Saturation
        self.saturation = True

        self.periodic = True

        self.file_lattice = file_lattice

        if file_lattice != "":
            try:
                Atom_pos, Atom_types, Lims, mass_map, type_map = read_structure_from_file(file_lattice, do_scale=True, atom_style="atom")
                self.Pos_lattice = Atom_pos
                self.Types_lattice = Atom_types
                self.Lims_lattice = Lims
                self.type_map = type_map
                self.mass_map = mass_map

            except:
                print("Tried reading the lattice file but an unknown issue happened.")


    @property
    def metal(self):
        """LAMMPS type ids mapped to metallic elements."""
        return tuple(
            type_id for type_id, symbol in self.type_map.items()
            if isinstance(symbol, str) and symbol.capitalize() in _METAL_ELEMENTS
        )

    @property
    def oxide_anion_types(self):
        """Type ids mapped to oxygen."""
        return tuple(
            type_id for type_id, symbol in self.type_map.items()
            if isinstance(symbol, str) and symbol.capitalize() == "O"
        )


    @property
    def oxide_cation_types(self):
        """Type ids mapped to non-oxygen, non-hydrogen oxide species."""
        return tuple(
            type_id for type_id, symbol in self.type_map.items()
            if isinstance(symbol, str)
            and symbol.capitalize() not in {"O", "H"}
        )

    @property
    def hydrogen_types(self):
        """Type ids mapped to hydrogen."""
        return tuple(
            type_id for type_id, symbol in self.type_map.items()
            if isinstance(symbol, str) and symbol.capitalize() == "H"
        )

    @property
    def material(self):
        """Infer whether the atom type map describes an oxide, metal, or other system."""
        if self.oxide_anion_types:
            return "oxide"
        symbols = {
            str(symbol).capitalize()
            for symbol in self.type_map.values()
            if isinstance(symbol, str)
        }
        if symbols and symbols <= _METAL_ELEMENTS:
            return "metal"
        return "unknown"


    def set_parameters(self, pitch, width, thickness, diameter, turns):
        """
        set_parameters(pitch, width, thickness, diameter, turns)

        This method sets the parameters of the helix

        Parameters
        ----------
            pitch : float
                the pitch, or the period of the helix
            width : float
                the width of the helix
            thickness : float
                the thickness of the helix
            diameter : float
                the diameter of the circle containing the helix.
            turns : float
                the amount of turns made by the helix

        Returns
        -------
            None
                The attributes are set
        """
        self.pitch = pitch
        self.width = width
        self.thickness = thickness
        self.diameter = diameter
        self.turns = turns

    def set_cast(self,int_thick):
        """
        set_cast(int_thick)

        Convert the helix as to put a cast inside.
        If the thickness is put at 0 (or below), then no cast are made

        Parameters
        ----------
            int_thick : float
                the thickness of the external layer around the cast

        Returns
        -------
            None
                The values are stored inside the helix object
        """
        if int_thick > 0:
            self.cast = True
            self.int_thick = int_thick
        else: self.castt = False


    def set_twist(self,twist):
        """
        set_twist(twist)

        Convert the helix into a twist (True) or not (False)

        Parameters
        ----------
            twist : bool

        Returns
        -------
            None
        """

        self.twist = twist

    def set_atoms(self,Pos,Types,Lims,Bonds,Angles,type_map,mass_map):
        """
        set_atoms(Pos,Types,Lims,Bonds,Angles)

        Set the atoms of the helix

        Parameters
        ----------
            Pos : list
                The position of the atoms
            Types : list
                The types of the atoms
            Lims : list
                The limit coordinates of the system
            Bonds : list
                The list of the bonds typically used for analysis
            Angles : list
                The angles of the OH bonds used to complete the system
            type_map : dict
                the mapping of the type of atom used by LAMMPS and the atomic name
            mass_map : dict
                the mapping of the type of atom used by LAMMPS and the atomic mass

        Returns
        -------
            None
                The attributes are set
        """
        self.Pos = Pos
        self.Types = Types
        self.Lims = Lims
        self.Bonds = Bonds
        self.Angles = Angles
        self.type_map = type_map
        self.mass_map = mass_map


    def set_saturation(self,saturation):
        """
        set_saturation(saturation)

        Set if the algorithm to saturate the system with H2O is done or not

        Parameters
        ----------
            saturation : bool

        Returns
        -------
            None
                The attribute is set
        """

        self.saturation = saturation

    def set_periodic(self,periodic):
        """
        set_periodic(periodic)

        Set if the system is periodic or not

        Parameters
        ----------
            periodic : bool

        Returns
        -------
            None
                The attribute is set
        """

        self.periodic = periodic

    def set_lattice(self,Pos_lattice,Types_lattice,Lims_lattice):
        """
        set_lattice(Pos_lattice,Types_lattice,Lims_lattice)

        Sets the parameters for the lattice

        Parameters
        ----------
            Pos_lattice : list
                The position of the atoms for the lattice
            Types_lattice : list
                The types of the atoms for the lattice
            Lims_lattice : list
                The limit coordinates for the lattice

        Returns
        -------
            None
                The attributes are set
        """
        self.Pos_lattice = Pos_lattice
        self.Types_lattice = Types_lattice
        self.Lims_lattice = Lims_lattice


    def read_lattice(self,file_lattice):
        """
        read_lattice(file_lattice)

        Reads a lattice file

        Parameters
        ----------
            file_lattice : str
                The name of the file containing the lattice

        Returns
        -------
            None
                The file is read and the attributes are set
        """
        self.file_lattice = file_lattice
        Atom_pos, Atom_types, Lims, mass_map, type_map = read_structure_from_file(file_lattice, do_scale=True, atom_style="atom")
        self.Pos_lattice = Atom_pos
        self.Types_lattice = Atom_types
        self.Lims_lattice = Lims
        self.type_map = type_map
        self.mass_map = mass_map

    def read_structure(self,file):
        """
        read_structure(file)

        Reads a data file for the helix

        Parameters
        ----------
            file : str
                The name of the file

        Returns
        -------
            None
                The file is read and the attributes Pos, Types and Lims are set
        """
        Atom_pos, Atom_types, Lims, mass_map, type_map = read_structure_from_file(file, do_scale=True, atom_style="atom")
        self.Pos = Atom_pos
        self.Types = Atom_types
        self.Lims = Lims
        if type_map != {}:
            self.type_map = type_map
        if mass_map != {}:
            self.mass_map = mass_map
