LAMMPS
======

Now that the helix has been created, the relaxation can be made using a MD software.
The one we support is `LAMMPS <https://lammps.org/>`_, but it should be possible to use any other software by converting the data files and setting the input files properly.



Dynamic
-------

One can use the template provided in the section :doc:`examples<file_examples>` to run the LAMMPS dynamic.
If there is no cast, the lammps file consists in the following steps::

    #. Loading and setting up the data
    #. Minimisation of energy
    #. Slow gradient from a low temperature (usually 300K) to a high temperature (usually 2000K)
    #. Relaxation of the system for a long time in NVT
    #. Slow relaxation of the system in NPT
    #. Relaxation of the system for a long time in NPT
    #. Slow gradient from the high temperature to 0.1K
    #. Minimisation of energy

If there is a cast, it consists in::

    #. Loading and setting up the data
    #. Minimisation of energy
    #. Slow gradient from a low temperature (usually 300K) to a high temperature (usually 2000K)
    #. Relaxation of the system for a long time in NVT
    #. Slow gradient from high to low temperature
    #. The cast is removed
    #. Slow gradient from low to high temperature
    #. Relaxation of the system for a long time in NVT
    #. Slow relaxation of the system in NPT
    #. Relaxation of the system for a long time in NPT
    #. Slow gradient from the high temperature to 0.1K
    #. Minimisation of energy


Output
------

After the LAMMPS computation, one may want to watch the dynamic, to do that one can use a software such as `VMD <https://www.ks.uiuc.edu/Research/vmd/>`_.
In SHAnC however, one needs to only work on the last timestep. As files can get extremely big, it is not recommended to load the entire file. In order to extract only the last timestep, one can use the following awk command ::

    awk '/^ITEM: TIMESTEP/ { block = $0 "\n"; next } { block = block $0 "\n" } END { printf "%s", block } ' FILENAME.lammpstrj > FILENAME_LAST_TIMESTEP.lammpstrj
