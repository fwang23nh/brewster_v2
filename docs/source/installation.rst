.. Brewster_v2 documentation master file, created by
   sphinx-quickstart on Tue Jan 27 13:08:34 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Installation
============


Python and Fortran dependencies
-------------------------------

Brewster_v2 depends on:

- Python 3.8 or higher
- Fortran compiler (gfortran)
- Python packages in ``requirements.txt``

Example
~~~~~~~

.. code-block:: bash

   git clone https://github.com/fwang23nh/brewster_v2.git
   cd brewster_v2
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   # build Fortran modules
   ./build

You can use ``check_brewster.py`` to test your installation.


Download Brewster v2 reference data
-----------------------------------

Download the Linelists and Cloud data from `Dropbox`_.

Make sure that the **Linelists** and **Cloud** folders are located at the same level as the ``brewster_v2`` folder.

.. _Dropbox: https://Dropbox.org/record/XXXXXXXX



