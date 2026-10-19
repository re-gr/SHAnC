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

__version__ = "0.1.0"
