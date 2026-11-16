Chemical Analysis
=================

Reminder:
    Do not forget to only work with the last timestep. One can extract it using the following awk command::

        awk '/^ITEM: TIMESTEP/ { block = $0 "\n"; next } { block = block $0 "\n" } END { printf "%s", block } ' FILENAME.lammpstrj > FILENAME_LAST_TIMESTEP.lammpstrj


Loading the relaxed helix
-------------------------

After extracting the last timestep, one can load it using the function :doc:`shanc.create_helix_from_file<../api/functions/create_helix_from_file>` or the method :doc:`Helix.read_structure<../api/methods/read_structure>`.
It is recommended to also load the starting helix as a lattice as it will provide all the mapping information easily.::

    import shanc
    import shanc.examples

    file_lattice = shanc.examples.SiO2_helix
    file_last_timestep = shanc.examples.SiO2_last_timestep

    pitch = 100.
    width = 30.
    thickness = 20.
    diameter = 60.
    turns = 1

    helix = shanc.Helix(pitch=pitch,width=width,thickness=thickness,diameter=diameter,turns=turns,file_lattice=file_lattice)
    helix.read_structure(file_last_timestep)


Plotting Methods
----------------

In order to visualize the last timestep, one can just use the method :doc:`Helix.plot_structure<../api/methods/plot_structure>`.
The color of each atom will by default be an association made with the atomic number, but one can change colors using the *type_color* argument.
If there is a cast in the helix, it will also be plotted.
It will produce the following result.::

    helix.plot_structure()

.. pyvista-plot::
    :include-source: false

    import shanc
    import shanc.examples
    import numpy as np

    file_lattice = shanc.examples.SiO2_helix
    file_last_timestep = shanc.examples.SiO2_last_timestep

    helix = shanc.Helix(file_lattice=file_lattice)
    helix.read_structure(file_last_timestep)

    helix.plot_structure()

::

    helix.plot_structure(draw_bonds = True, do_fancy = True)

.. pyvista-plot::
    :include-source: false

    import shanc
    import shanc.examples

    file_lattice = shanc.examples.SiO2_helix
    file_last_timestep = shanc.examples.SiO2_last_timestep

    helix = shanc.Helix(file_lattice=file_lattice)
    helix.read_structure(file_last_timestep)
    helix.plot_structure(do_fancy=True,draw_bonds=True)

One has to remark that computing the bonds for the whole system and doing a "fancier" plot can take a bit longer to plot.
Also, due to the periodicity, some bonds crosses the whole system. One can remove this behaviour by using the :doc:`Helix.set_periodic<../api/methods/set_periodic>` method.


Radial Distribution functions
-----------------------------

It is possible to plot radial distribution functions using :doc:`helix.plot_rdf<../api/methods/plot_rdf>`.
The selection of the atoms can be made using the first two arguments. They can be either integers or list of integers to select types. Or it can also be strings such as "O,H" to represent multiple atoms.


.. plot::
    :include-source: true
    :show-source-link: false

    import shanc
    import shanc.examples

    file_lattice = shanc.examples.SiO2_helix
    file_last_timestep = shanc.examples.SiO2_last_timestep

    helix = shanc.Helix(file_lattice=file_lattice)
    helix.read_structure(file_last_timestep)
    helix.plot_rdf("Si","O")


Coordination number
-------------------

Using the bonds, the coordination number of each atom can be computed using :doc:`helix.plot_coordination<../api/methods/plot_coordination>`.
This will produce the following graph:


.. plot::
    :context: reset

    import shanc
    import shanc.examples

    file_lattice = shanc.examples.SiO2_helix
    file_last_timestep = shanc.examples.SiO2_last_timestep

    helix = shanc.Helix(file_lattice=file_lattice)
    helix.read_structure(file_last_timestep)
    helix.plot_coordination("Si")

It is also possible to decompose the coordination number each atom type using the *decompose* keyword.

.. plot::
    :context: reset

    import shanc
    import shanc.examples

    file_lattice = shanc.examples.Ag2GePbS4
    pitch = 40.
    width = 20.
    thickness = 20.
    diameter = 30
    turns = 1

    helix = shanc.Helix(pitch=pitch,width=width,thickness=thickness,turns=turns,file_lattice=file_lattice)
    helix.create_helix()
    helix.plot_coordination(1,decompose=True)
