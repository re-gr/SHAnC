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