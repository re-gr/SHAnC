Formats of files
================

.data
-----

The recommended format for data files is the data format from `lammps <https://docs.lammps.org/Run_formats.html#data-file>`_::

    A atoms
    B atom types

    XMIN XMAX xlo xhi
    YMIN YMAX xlo xhi
    ZMIN ZMAX xlo xhi

    Masses

    1 M1
    ...

    Atoms

    1 T1 C X1 Y1 Z1
    ...


This file has the advantages of containing the masses (M1...) of the atoms and the size of the box (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX).


.xyz
----
It is however possible to open other type of files.
If one were to use xyz files, an automatic mapping of the masses will be done and a box will be created.
It is however not recommended to use xyz files to create lattices and should only be used for other purposes such as loading the chromophores.
