import shanc
import shanc.examples

file_lattice = shanc.examples.SiO2
pitch = 100.
width = 30.
thickness = 20.
diameter = 60.
turns = 1

helix = shanc.helix(pitch=pitch,width=width,thickness=thickness,diameter=diameter,turns=turns,file_lattice=file_lattice)
helix.create_helix()
helix.plot_structure()