Analysis of the helices
=======================

Reminder:
    Do not forget to only work with the last timestep. One can extract it using the following awk command:
        awk '/^ITEM: TIMESTEP/ { block = $0 "\n"; next } { block = block $0 "\n" } END { printf "%s", block } ' FILENAME.lammpstrj > FILENAME_LAST_TIMESTEP.lammpstrj


Loading the relaxed helix
-------------------------

After extracting the last timestep, one can load it using the function :doc:`shanc.create_helix_from_file<../api/functions/create_helix_from_file>` or the method :doc:`Helix.read_structure<../api/methods/read_structure>`

    import shanc
    import shanc.examples

    file_lattice = shanc.examples.SiO2
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
It will produce the following result.


helix.plot_structure()

.. pyvista-plot::
    :include-source: false

    import shanc
    import shanc.examples
    import numpy as np

    file_lattice = shanc.examples.SiO2
    file_last_timestep = shanc.examples.SiO2_last_timestep

    helix = shanc.Helix(file_lattice=file_lattice)
    helix.read_structure(file_last_timestep)

    helix.plot_structure()


helix.plot_structure(draw_bonds = True, do_fancy = True)

.. pyvista-plot::
    :include-source: false

    import shanc
    import shanc.examples

    file_lattice = shanc.examples.SiO2
    file_last_timestep = shanc.examples.SiO2_last_timestep

    helix = shanc.Helix(file_lattice=file_lattice)
    helix.read_structure(file_last_timestep)
    helix.plot_structure(do_fancy=True,draw_bonds=True)

One has to remark that computing the bonds for the whole system and doing a "fancier" plot can take a bit longer to plot.
Also, due to the periodicity, some bonds crosses the whole system. One can remove this behaviour by using the :doc:`Helix.set_periodic<../api/methods/set_periodic>` method.
