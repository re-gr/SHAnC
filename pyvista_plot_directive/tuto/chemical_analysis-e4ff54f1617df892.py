import shanc
import shanc.examples
import numpy as np

file_lattice = shanc.examples.SiO2_helix
file_last_timestep = shanc.examples.SiO2_last_timestep

helix = shanc.Helix(file_lattice=file_lattice)
helix.read_structure(file_last_timestep)

helix.plot_structure(draw_bonds=True)