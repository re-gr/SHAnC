import shanc

def test_helix():
    """
    test to create and modify the helix
    """
    helix = shanc.Helix()

    pitch = 10
    width = 10
    thickness = 10
    diameter = 10
    turns = 1
    helix.set_parameters(pitch, width, thickness, diameter, turns)

    helix.set_cast(10)
    helix.set_twist(True)
    helix.set_saturation(True)
    helix.set_periodic(False)
    assert 1


def test_construction_helix():
    """
    test to create and do the distortion of an helix
    """
    pitch = 50
    width = 20
    thickness = 10
    diameter = 30
    turns = 1
    file_lattice = "shanc/demo_system/files_lattices/Model_file/beta_quartz.data"

    helix = shanc.Helix(pitch,width,thickness,diameter,turns,file_lattice)
    helix.create_helix()
    print(helix.material)
    print("Lims",helix.Lims)
    #helix.plot_structure_quick()
    assert 1

test_construction_helix()



def test_read():
    file_cuboid = "cuboid.data"
    helix = shanc.create_helix_from_file(file_cuboid, frame=-1, data_file=None, 
                             type_map={1: "Au"}, 
                             atom_style="atom")
    print("Lims",helix.Lims)
    print(helix.mass_map)
    print(helix.oxide_anion_types)
    print(helix.material)
    assert 1


def test_analysis_oxide():
    file_helix = "shanc/demo_system/SiO2/last_timestep.lammpstrj"
    helix = shanc.create_helix_from_file(file_helix, frame=-1, data_file=None, 
                                 type_map={1: "Si", 2:"O", 3:"O", 4:"H"}, 
                                 atom_style="atom")
    helix.plot_structure()
    helix.plot_rdf_cation_anion()
    helix.plot_coordination_oxide()
    assert 1


def test_analysis_metal(): 
    file_helix = "shanc/demo_system/Au/au_helix_relaxed.lammpstrj"
    helix = shanc.create_helix_from_file(file_helix, frame=-1, data_file=None, 
                                    type_map={1: "Au"}, 
                                    atom_style="atom")
    helix.plot_structure()
    helix.plot_rdf_metal()
    helix.plot_coordination_metal()
    assert 1



test_read()
test_analysis_oxide()
test_analysis_metal()



