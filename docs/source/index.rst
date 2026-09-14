.. Brewster_v2 documentation master file, created by
   sphinx-quickstart on Tue Jan 27 13:08:34 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Brewster v2
===========

Brewster_v2 is a spectral inversion and retrieval code for emission spectra
of brown dwarfs and giant planets. It combines a layered atmosphere, gas and
cloud opacity, radiative transfer, instrumental processing, and Bayesian
inference in one configurable workflow.

New users should start with :doc:`getting_started`, then choose the atmospheric
parameterizations described in :doc:`model_components/index`. The notebook
tutorials show complete workflows, while the API reference documents the
underlying Python interfaces.

Start here
----------

* :doc:`getting_started` — understand the v2 configuration objects and run a
  first model.
* :doc:`model_components/index` — compare P–T, gas-profile, chemistry, and
  cloud parameterizations.
* :doc:`tutorials` — follow spectrum generation, retrieval, analysis, and
  cloud diagnostics.
* :doc:`api` — look up Python functions and configuration classes.


.. toctree::
   :maxdepth: 2

   getting_started
   installation
   model_components/index
   retrieval_workflow
   tutorials
   postprocessing
   api
