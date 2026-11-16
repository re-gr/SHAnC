Multithreading
==============

Various functions, most notably the function used for analysis can be multithreaded.
By default, they are multithreaded for 4 processors.
Due to python shenanigans unfortunately putting more processors might not be faster...

One can change the number of processors using ::

    shanc.set_number_procs(4)
