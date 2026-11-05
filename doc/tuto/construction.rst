Construction of helices
=======================


In order to construct helices of a certain material, one needs a data file containing the lattice of the structure.
The recommended format for this is the data format from `LAMMPS <https://docs.lammps.org/Run_formats.html#data-file>`_

Before the construction of the helix, one should load and visualize the lattice by using the :doc:`Helix.plot_structure<../api/methods/plot_structure>` method.


Dimensions
----------

The dimensions that can be set are:

- **pitch**
- **width**
- **thickness**
- **diameter**
- **turns**


Twisted Ribbons
---------------
Instead of simulating helices, twisted ribbons can also be constructed.
This can be done by using the :doc:`Helix.set_twist<../api/methods/set_twist>` method.
The twisted ribbons look as follows:

.. pyvista-plot::

  import shanc
  import shanc.examples

  file_lattice = shanc.examples.SiO2
  pitch = 100.
  width = 30.
  thickness = 20.
  turns = 1

  helix = shanc.Helix(pitch=pitch,width=width,thickness=thickness,turns=turns,file_lattice=file_lattice)
  helix.set_twist(True)
  helix.create_helix()
  helix.plot_structure()


Cast inside the helix
---------------------

During the relaxation the helix, due to the constraints put by the distortions, the helix can collapse on itself.
In order to circumvent this issue, one solution consists on putting a cast inside the helix, that is, to freeze the atoms inside and relax a certain layer on it.
This can be done using the :doc:`Helix.set_cast<../api/methods/set_cast>` method:

    helix.set_cast(int_thickness=35.0)

The argument represents the thickness of the outside layer.

During the construction of the helix, the atoms that will be the cast will be saved in another file and one will need to configure the force field correctly for the cast to be effective.
More details can be found in the :doc:`lammps<lammps>` part.


Saturation of the system
------------------------

For systems containing oxides, it is possible to saturate the system using water.
HO groups will bond to unsaturated cations and Hydrogens will bond to existing Oxygens, converting them to bonding oxygens.
Two types of atoms are added that way, the Hydrogens and bonding Oxygens. One has to properly set the charge of the new types in the :doc:`lammps input file<lammps>`.


Construction of the helix
-------------------------

Now that the parameters have been set, one can construct the helix using the :doc:`shanc.create_helix<../api/methods/create_helix>` method.
This will create one or two files depending if a cast is present or not. One can change the name of these files using the "output_name" and "output_name_cast" keywords.

The following code produces a silica twisted ribbon that is saturated with water with a cast

.. pyvista-plot::

    import shanc
    import shanc.examples

    file_lattice = shanc.examples.SiO2
    pitch = 100.
    width = 30.
    thickness = 20.
    turns = 1

    helix = shanc.Helix(pitch=pitch,width=width,thickness=thickness,turns=turns,file_lattice=file_lattice)
    helix.set_twist(True)
    helix.set_saturation(True)
    helix.set_cast(5)

    helix.create_helix(output_name="SiO2_helix.data",output_name_cast="SiO2_helix_cast.data")
    helix.plot_structure()
