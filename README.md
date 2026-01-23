<h1 align="center">
SHAnC
</h1>

SHAnC (Silica Helices Analysis & Construction) is a collection of scripts used to Construct and Analyze Silica Helices.

- **Github :** https://github.com/re-gr/SHAnC

## Getting Started

- Construction : the scripts that are used to create an helix or a twisted ribbon
    - distorsion.py : create helices and ribbons
    - read\_write.py : read and write files
    - script\_analysis.py : analysis (rdf, saturation ...) of structures
    - script\_cycles.py : cycle analysis of structures (WIP)

- Torsion : the scripts are used to twist the cuboids to the proper shape. This method should not be used, but is kept here just in case.
    - distorsion.py : old version of the script in construction that should be compatible with the other script
    - script\_launch.py : script to create the systems and input file for the simulations

- Animation : scripts that were used to create animation for slides purposes

## Depedencies

	numpy
	scipy
   	matplotlib
	pyvista
	lammps (optional, for torsion only)

## Contact

Email : [Rg](mailto:remi.grincourt@ens-lyon.fr)


