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
  import shanc.examples

  pitch = 50.
  width = 30.
  thickness = 20.
  diameter = 40.
  turns = 1

  helix = shanc.helix(pitch=pitch,width=width,thickness=thickness,diameter=diameter,turns=turns,file_lattice=shanc.examples.SiO2)
  helix.create_helix()
  helix.plot_structure()
