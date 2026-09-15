Cloud and contribution post-processing
======================================

The repository module :mod:`cloud_postprocessing` provides v2-native cloud
diagnostics. It replaces the v1 ``cloud_props.py`` assumptions with the current
configuration objects and forward model.

Restore a retrieval
-------------------

.. code-block:: python

   import pickle
   import numpy as np
   import settings
   import utils

   flat_chain, flat_logp, ndim = utils.get_endchain(runname, 1, path)
   theta = flat_chain[np.argmax(flat_logp)]
   runargs = utils.pickle_load(path + runname + "_runargs.pic")
   opacities = utils.pickle_load(path + runname + "_opacities.pic")
   cloudata = utils.pickle_load(path + runname + "_cloudata.pic")

   with open(path + runname + "_configs.pic", "rb") as handle:
       configs = pickle.load(handle)

   re_params = configs["re_params"]
   settings.init(runargs)
   settings.linelist, settings.cia = opacities[:2]
   settings.cloudata = cloudata

Species-resolved cloud photospheres
-----------------------------------

.. code-block:: python

   from cloud_postprocessing import get_cloud_photospheres

   diagnostics = get_cloud_photospheres(theta, re_params, runargs)

   diagnostics.wavelength       # (nwave,)
   diagnostics.total_cloud      # (npatch, nwave)
   diagnostics.gas              # (npatch, nwave)
   diagnostics.species          # (npatch, ncloud, nwave)
   diagnostics.contribution     # (npatch, nwave, nlayers)
   diagnostics.labels

A zero in a photosphere curve means cumulative optical depth did not reach one
inside the model grid. The species calculation makes one diagnostic forward-
model call per cloud component, in addition to the total call.

Layer-by-layer optical thickness
--------------------------------

.. code-block:: python

   from cloud_postprocessing import (
       get_cloud_layer_optical_depth,
       plot_cloud_layer_optical_depth,
   )

   layers = get_cloud_layer_optical_depth(theta, re_params, runargs)

   # Differential Delta-tau in each pressure layer.
   plot_cloud_layer_optical_depth(layers, patch=0, cumulative=False)

   # Tau integrated downward from the atmosphere top.
   plot_cloud_layer_optical_depth(layers, patch=0, cumulative=True)

``layers.optical_depth`` and ``layers.cumulative_optical_depth`` both have
shape ``(npatch, ncloud, nlayers)``. They are evaluated at Brewster's 1 µm
cloud-reference wavelength.

Runnable notebook
-----------------

See :doc:`tutorials/Brewster_v2_Tutorial_6_Cloud_Postprocessing` for the
combined contribution-function plot and individual cloud curves.

Gas abundances and equilibrium chemistry
----------------------------------------

See :doc:`tutorials/Brewster_v2_Tutorial_7_Gas_Abundances_vs_Chemical_Equilibrium`
to derive C/O and [M/H] from a free-chemistry posterior, evaluate the chemical-
equilibrium and BFF grids along the retrieved P–T profile, calculate the gas
photosphere, and compare the retrieved and equilibrium abundance profiles.
