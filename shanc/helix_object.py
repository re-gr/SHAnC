"""
Helix object definition for SHAnC
"""


from shanc.read_write import *



class Helix:
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
        bonds_mapping : dict
            the mapping of the number of bonds a type of atom should have
        threshold_map : dict
            the mapping for the threshold used to compute bonds
        cast : bool
            If a cast is present or not.
        int_thick : float
            The thickness of the outside layer
        Pos_cast : list
            The position of the atoms of the cast
        Types_cast : list
            The type of the atoms of the cast
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
        Bonds_analysis : list
            The array containing the bonds computed by SHAnC
        Coordination : list
            The array containing the coordination distribution
        rdf : list
            The array containing the RDF of every type with the others

    Parameters
    ----------
        None
            This is just an initialisation

    Returns
    -------
        helix : helix
            The Helix object.
    """

    def __init__(self, pitch = 0., width = 0., thickness = 0., diameter = 0., turns = 1, file_lattice = ""):
        """
        Initialisation function
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
        self.bonds_mapping = {}
        self.threshold_map = {}

        #Values cast
        self.cast = False
        self.int_thick = 0.
        self.Pos_cast = []
        self.Types_cast = []

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

        self.Bonds_analysis = []
        self.Coordination = []
        self.rdf = []

        if file_lattice != "":
            try:
                Atom_pos, Atom_types, Lims, mass_map, type_map = read_structure_from_file(file_lattice, do_scale=True, atom_style="atom")
                self.Pos_lattice = Atom_pos
                self.Types_lattice = Atom_types
                self.Lims_lattice = Lims
                self.type_map = type_map
                self.mass_map = mass_map
                if not "O" in type_map.values():
                    print("No oxygens detected in the system, will not try to saturate the system with H2O")
                    self.saturation = False


            except:
                print("Tried reading the lattice file but an unknown issue happened.")


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


    def set_type_map(self,type_map):
        """
        set_type_map(type_map)

        Sets the mapping for the type and atomic symbol

        Parameters
        ----------
            type_map : dict
                the type mapping

        Returns
        -------
            None
        """
        self.type_map = type_map

    def set_mass_map(self,mass_map):
        """
        set_mass_map(mass_map)

        Sets the mapping for the mass of each type

        Parameters
        ----------
            mass_map : dict
                the mass mapping

        Returns
        -------
            None
        """
        self.mass_map = mass_map

    def set_bonds_mapping(self,bonds_mapping):
        """
        set_bonds_mapping(bonds_mapping)

        Sets the mapping for the count of bonds.
        This is used when trying to plot the defects of the system.
        One can set a list for multiple acceptable number of bonds.

        Parameters
        ----------
            bonds_mapping : dict
                the bonds mapping

        Returns
        -------
            None
        """
        self.bonds_mapping = bonds_mapping

    def set_threshold_map(self,threshold_map):
        """
        set_threshold_map(threshold_map)

        Sets the mapping for the threshold map.
        The threshold is used to compute the bonds of the system and should be assigned for every type

        Parameters
        ----------
            threshold_map : dict
                the threshold_map mapping

        Returns
        -------
            None
        """
        self.threshold_map = threshold_map


def create_helix_from_file(file, do_scale=True, atom_style="atom", type_map = None, mass_map = None):
    """
    read_structure_from_data(file, do_scale=True, atom_style="full")

    This function tries to read a structure file from a file.
    The format of the file should be .data, .xyz or .lmmpstrj

    SHould be used to display after dynamics

    """

    Atom_pos, Atom_types, Lims, parsed_mass_map, parsed_type_map = read_structure_from_file(file, do_scale=do_scale, atom_style=atom_style)

    helix_created = Helix()

    resolved_type_map = parsed_type_map if type_map is None else type_map
    resolved_mass_map = parsed_mass_map if mass_map is None else mass_map

    helix_created.set_atoms(Atom_pos,Atom_types,Lims,[],[], type_map=resolved_type_map, mass_map=resolved_mass_map)

    return helix_created
