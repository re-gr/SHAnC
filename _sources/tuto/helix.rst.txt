Helix Object
============

The helix object contains numberous attributes and methods.
The list can be found :doc:`here<../api/helix>`.

Lists
-----

There are three types of lists containing the atomic information,

.. list-table::

    * - Lattice : Pos_lattice, Types_lattice, Lims_lattice
    * - Helix : Pos, Types, Lims
    * - Cast : Pos_cast, Types_cast

The Pos lists contain the position of the atoms, Types suprisingly contain the types and Lims contain the boundaries.




Dictionnaries
-------------

Various attributes of the Helix object are dictionnaries. They are the following,


.. list-table::

    * - type_map : mapping between the atom type (int) and the atomic symbol (str)
      - this mapping is used to automatically attribute colors and radius to atoms, but also to select atom types easily, such as for :doc:`plotting rdf<../api/api/methods/plot_rdf>`
    * - mass_map : mapping between the atom type (int) and their masses (float)
      - this mapping is used to automatically detect the atomic symbol (if close to a known one) and to create data files properly
    * - threshold_map : mapping between the atom type (int) and the threshold for bonds (float)
      - this mapping is used to compute the bonds of the system using a threshold specific for each atom type
    * - bonds_mapping : mapping between the atom type (int) and the nmuber of bonds they should have (int)
      - this is used when trying to compute the :doc:`coordination of atoms<../api/methods/plot_coordination>`

Most of these objects are either automatically created when loading files (typically type_map and mass_map from loading data files), or have default values for atomic symbol (threshold_map and bonds_mapping).
It is however possible to manually change these values using the corresponding methods.
