Helix Object
============

Attributes
----------

.. list-table::

    * - pitch : float
      - the pitch, or the period of the helix
    * - width : float
      - the width of the helix
    * - thickness : float
      - the thickness of the helix
    * - diameter : float
      - the diameter of the circle containing the helix.
    * - turns : float
      - the amount of turns made by the helix
    * - Pos : list
      - The position of the atoms
    * - Types : list
      - The types of the atoms
    * - Lims : list
      - The limit coordinates of the system
    * - Bonds : list
      - The list of the bonds typically used for analysis
    * - Angles : list
      - The angles of the OH bonds used to complete the system
    * - type_map : dict
      - the mapping of the type of atom used by LAMMPS and the atomic name
    * - mass_map : dict
      - the mapping of the type of atom used by LAMMPS and the atomic mass
    * - bonds_mapping : dict
      - the mapping of the number of bonds a type of atom should have
    * - threshold_map : dict
      - the mapping for the threshold used to compute bonds
    * - cast : bool
      - If a cast is present or not.
    * - int_thick : float
      - The thickness of the outside layer
    * - Pos_cast : list
      - The position of the atoms of the cast
    * - Types_cast : list
      - The type of the atoms of the cast
    * - Pos_lattice : list
      - The position of the atoms for the lattice
    * - Types_lattice : list
      - The types of the atoms for the lattice
    * - Lims_lattice : list
      - The limit coordinates for the lattice
    * - twist : bool
      - If the helix is a twisted ribbon or not
    * - saturation : bool
      - If the helix has to be saturated or not
    * - periodic : bool
      - If the system should be treated as periodic or not
    * - file_lattice : str
      - The name of the file containing the lattice.
    * - Bonds_analysis : list
      - The array containing the bonds computed by SHAnC
    * - Coordination : list
      - The array containing the coordination distribution
    * - rdf : list
      - The array containing the RDF of every type with the others




Methods
-------


.. list-table::

    * - :doc:`Helix<methods/helix>`
      - create the helix object
    * - :doc:`Helix.set_parameters<methods/set_parameters>`
      - set the construction parameters of the helix
    * - :doc:`Helix.set_lattice<methods/set_lattice>`
      - set the lattice of the system
    * - :doc:`Helix.read_lattice<methods/read_lattice>`
      - read a file as a lattice for the helix
    * - :doc:`Helix.set_cast<methods/set_cast>`
      - set the thickness of the external layer and thus putting a cast inside
    * - :doc:`Helix.set_twist<methods/set_twist>`
      - set the construction as a twist or not
    * - :doc:`Helix.set_saturation<methods/set_saturation>`
      - set if the system should be saturated with water or not
    * - :doc:`Helix.set_periodic<methods/set_periodic>`
      - set if the system should be considered as periodic or not
    * - :doc:`Helix.set_atoms<methods/set_atoms>`
      - set the atoms of the helix
    * - :doc:`Helix.read_structure<methods/read_structure>`
      - reads the structure of the helix from a file
    * - :doc:`Helix.set_type_map<methods/set_type_map>`
      - set the mapping for type and atomic symbol
    * - :doc:`Helix.set_mass_map<methods/set_mass_map>`
      - set the mapping for type and mass
    * - :doc:`Helix.set_bonds_mapping<methods/set_bonds_mapping>`
      - set the mapping for the number of bonds for each type
    * - :doc:`Helix.set_threshold_map<methods/set_threshold_map>`
      - set the mapping of the threshold used to compute the bonds and each type

Creation Methods
----------------

.. list-table::

    * - :doc:`shanc.create_helix_from_file<functions/create_helix_from_file>`
      - creates an helix object from a file


.. toctree::
    :hidden:

    methods/helix
    methods/set_parameters
    methods/set_lattice
    methods/read_lattice
    methods/set_cast
    methods/set_twist
    methods/set_saturation
    methods/set_periodic
    methods/set_atoms
    methods/read_structure
    methods/set_type_map
    methods/set_mass_map
    methods/set_bonds_mapping
    methods/set_threshold_map
    functions/create_helix_from_file
