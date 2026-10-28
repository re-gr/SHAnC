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
    file_lattice = "demo_system//Analysis_example//sio2_pitch_200/beta_quartz.data"

    helix = shanc.Helix(pitch,width,thickness,diameter,turns,file_lattice)
    helix.create_helix()
    print(helix.material)
    print("Lims",helix.Lims)
    helix.plot_structure_quick()
    assert 1

test_construction_helix()
