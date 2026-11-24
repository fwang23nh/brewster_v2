.. Brewster_v2 documentation master file, created by
   sphinx-quickstart on Tue Jan 27 13:08:34 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Installation
============


Python and Fortran dependencies
-------------------------------

Brewster_v2 depends on:

- Python 3.11 for the environment example below. Support for other versions
  should be checked separately; the repository README and earlier documentation
  give conflicting version guidance.
- Fortran compiler (gfortran)
- Python packages in ``requirements.txt``
- gas opacity data and cloud optical data to be downloaded separately.

Example
~~~~~~~

.. code-block:: bash

   git clone https://github.com/fwang23nh/brewster_v2.git
   cd brewster_v2
   conda create -n brewster python=3.11
   conda activate brewster
   conda install --file requirements.txt -c conda-forge
   # build Fortran modules
   make clean
   ./build

This example uses Conda throughout; run the build and subsequent Python
commands in the same environment. The dependency list includes native runtime
and MPI packages as well as Python packages. Install a compatible Fortran
compiler before building. MultiNest retrievals also require PyMultiNest and its
native MultiNest library; these are not included in ``requirements.txt``.
The commands above are an installation recipe. Every ``./build`` prints the
platform, Python interpreter, NumPy and f2py versions, Fortran compiler, and
build tool versions, including when the extensions are already up to date.
When reporting a build failure, include that diagnostics block and the error
output.

Fortran sources live in ``forward_model/``. The build writes the four Python
extensions, Fortran objects and modules, static library, and generated f2py
signatures to ``forward_model/build/``. Use ``./build PYTHON=/path/to/python``
to select a specific Python interpreter. If upgrading an existing checkout,
move any old root-level extension binaries out of the repository before
using the new build.

The default CIA and chemical-equilibrium tables are resolved from the
repository's ``data/`` directory. Custom paths supplied in a driver are
interpreted as before.


Download Brewster v2 reference data
-----------------------------------

You will need to download the gas opacity data files and cloud optical data, and place them in folders accessible from your Brewster working folder. We have called these folders "Linelists" and "Clouds" in examples and tutorials, but they can have any name.  The default location for these is assumed to be on the same level as the ``brewster_v2`` folder, but can be anywhere and you are able to set this path in driver files.


`Download R = 10K opacity files`_

`Download R = 30K opacity files`_

`Download R = 100K opacity files`_

.. _Download R = 10K opacity files: https://star.herts.ac.uk/~bb/public_files/R10K_Feb2026.tar.gz
.. _Download R = 30K opacity files: https://star.herts.ac.uk/~bb/public_files/R30K_Jan2026.tar.gz
.. _Download R = 100K opacity files: https://star.herts.ac.uk/~bb/public_files/R100K_Mar2026.tar.gz


`Download Cloud Mie Coefficient Files`_

.. _Download Cloud Mie Coefficient Files: https://star.herts.ac.uk/~bb/public_files/clouds.tar.gz


Once these are downloaded and set up you can use ``check_brewster.py`` to test your installation and the build.

(You will need to update the paths at the start of the file to point at your gas opacity folder and clouds folder.)

.. code-block:: bash

   python check_brewster.py

The check runs clear and cloudy benchmark cases and reports whether their
outputs differ by more than 1% from stored benchmarks. It uses
``data/gaslistR10K_old.dat``: make sure the downloaded files match that manifest.
Passing this comparison establishes agreement with those benchmarks, not
independent physical validation. Investigate reported differences before
changing reference outputs.

For a worked spectrum-generation example, follow :doc:`quickstart`, which
links to Tutorial 2 and lists its required gas and cloud data.

If building or loading data fails, see :doc:`troubleshooting`. For the framework
papers, see :doc:`citation_releases_contributing`.
