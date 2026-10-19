"""
Helix object definition for SHAnC
"""


from shanc.read_write import *


class helix:
    """
    Helix object
    """

    def __init__(self, pitch = 0., width = 0., thickness = 0., diameter = 0., turns = 1, file_lattice = ""):
        """
        helix(pitch, width, thickness, diameter, turns)

        Constructs the helix object containing the atoms and the casts.
        This is just the initialisation, please use helix.set_parameters ...

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
                The coordinate limits of the system
            Bonds : list, optional
                The list of the bonds typically used for analysis
            Angles : list
                The angles of the OH bonds used to complete the system
            Pos_cast : list, optional
                The position of the atoms of the cast
            Types_cast : list, optional
                The type of the atoms of the cast
            Lims_cast : list, optional
                The coordinate limits of the cast

        Parameters
        ----------
            None
                This is just an initialisation

        Returns
        -------
            helix : helix
                The helix object.
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

        Convert the helix into a twist (true) or not

        Parameters
        ----------
            twist : bool

        Returns
        -------
            None
        """

        self.twist = twist

    def set_atoms(self,Pos,Types,Lims,Bonds,Angles,type_atom_map):
        """
        set_atoms(Pos,Types,Lims,Bonds,Angles)

        Set the atoms of the helix

        """
        self.Pos = Pos
        self.Types = Types
        self.Lims = Lims
        self.Bonds = Bonds
        self.Angles = Angles
        self.type_atom_map = type_atom_map


    def set_saturation(self,saturation):
        """
        set_saturation(saturation)

        Set if the algorithm to saturate the system with H2O is done or not
        """

        self.saturation = saturation

    def set_periodic(self,periodic):
        """
        set_periodic(periodic)

        Set if the system is periodic or not
        """

        self.periodic = periodic

    def set_lattice(self,Pos_lattice,Types_lattice,Lims_lattice):
        """
        set_lattice(Pos_lattice,Types_lattice,Lims_lattice)

        Sets the parameters for the lattice
        """
        self.Pos_lattice = Pos_lattice
        self.Types_lattice = Types_lattice
        self.Lims_lattice = Lims_lattice


    def read_lattice(self,file_lattice):
        """
        read_lattice(file_lattice)

        reads a lattice file
        """
        self.file_lattice = file_lattice
        Atom_pos, Atom_types, Lims, mass_map, type_map = read_structure_from_file(file_lattice, do_scale=True, atom_style="atom")
        self.Pos_lattice = Atom_pos
        self.Types_lattice = Atom_types
        self.Lims_lattice = Lims
        self.type_map = type_map
        self.mass_map = mass_map
