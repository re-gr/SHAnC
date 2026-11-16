"""
Importation of the requiered libraries
"""

import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from matplotlib.widgets import Slider
import scipy as sp
import scipy.spatial.distance as sd
import scipy.signal as sps
import matplotlib.ticker as ticker
import re
from collections import defaultdict
import os
import time
from pathlib import Path
from collections import Counter

from shanc.helix_object import *
from shanc.analysis import *
from shanc.read_write import *
# import shanc.cycle_extraction
from shanc.add_chromophore import *

# Core numerical and geometry utilities
# - numpy: array maths and reshaping
# - scipy.spatial.KDTree: fast neighbour searches for atom pairs
# - scipy.spatial.distance: distance matrices and pairwise distances
# - scipy.signal: signal and histogram processing
# - matplotlib: plotting and interactive widgets
# - pyvista: 3D structure rendering and isosurfaces
# - re: parsing text in XYZ/LAMMPS file headers
# - defaultdict: convenient dictionary defaults for counters and maps

__version__ = "0.0.1.3"


nprocs = 4

def set_number_procs(nprocs):
    """
    set_number_procs

    This function sets the number of processor that will be used.
    Due to python shenanigans, one should use at most 8.
    Higher number of processors will unfortunately usually slow down calculations
    In my experience, between 4 and 8 processors no real differences arise.


    Parameters
    ----------
        nprocs : int
            the maximum number of processors

    Returns
    -------
        None
    """

    if type(nprocs) is int:
        shanc.nprocs = nprocs
    else:
        raise TypeError('The number of processors used should be an integer. You gave a {}'.format(type(nprocs)))

set_nprocs = set_number_procs
