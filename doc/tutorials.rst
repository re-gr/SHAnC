Tutorials
=========


Getting Started
---------------

* :doc:`Installation<tuto/installation>`


.. toctree::
  :hidden:

  tuto/installation

.. pyvista-plot::

  import pyvista as pv
  mesh = pv.Sphere()
  mesh.plot()


.. pyvista-plot::

  import shanc
  import numpy as np


  target_folder = '/home/rg/SHAnC_2/SHAnC/demo_system/Analysis_example/sio2_pitch_200'

  file = target_folder + "/last_timestep.lammpstrj"
  list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS = c.read_write.read_dump(file,unscale=True)

  # Analyze data file
  Pos = list_ATOMS[-1][:,2:]
  Types = list_ATOMS[-1][:,1]
  Lims = list_BOX[-1]

  # Plots and visualisation of the structure
  shanc.analysis.plot_analyze_defects(Pos,Types,periodic=True,Lims=Lims)

