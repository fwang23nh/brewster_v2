.. Brewster_v2 documentation master file, created by
   sphinx-quickstart on Tue Jan 27 13:08:34 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Installation
============


Python and Fortran dependencies
-------------------------------

Brewster_v2 depends on:

- Python 3.11 or higher (tested - may work on ealier versions YMMV)
- Fortran compiler (gfortran)
- Python packages in ``requirements.txt``
- gas opacity data and cloud optical data to be downloaded separately.

Example
~~~~~~~

.. code-block:: bash

   git clone https://github.com/fwang23nh/brewster_v2.git
   cd brewster_v2
   python3 -m venv venv
   source venv/bin/activate
   conda install --file requirements.txt -c conda-forge
   # build Fortran modules
   make clean
   ./build


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
