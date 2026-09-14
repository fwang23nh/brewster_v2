Retrieval workflow
==================

1. Choose the model
-------------------

Select the P–T, gas, chemistry, cloud, patch, and sampler options. Construct
``Retrieval_params`` and inspect ``re_params.dictionary``. Parameter ordering
in that dictionary is the ordering used by the posterior vector.

2. Configure data and opacities
-------------------------------

Construct ``ModelConfig`` with the observation, distance, line-list path,
cloud-opacity path, and instrument behaviour. This builds the pressure and
wavenumber grids and loads the required opacity tables.

3. Check the forward model
--------------------------

Run the spectrum-generation tutorial before launching a sampler. Confirm that
the model covers the data wavelength range, flux units match, the convolution
is appropriate, and every parameter vector produces finite output.

4. Choose a sampler
-------------------

``samplemode="mcmc"`` uses the ensemble-MCMC workflow and retrieves ``logg``
and ``r2d2`` directly. ``samplemode="multinest"`` uses nested sampling and its
mass/radius/distance parameterization. The associated default priors are stored
in different dictionary fields; always inspect them rather than assuming the
two modes are identical.

5. Save reproducibility products
--------------------------------

A complete analysis needs the chain or posterior samples together with:

* ``*_runargs.pic`` — numerical model inputs;
* ``*_configs.pic`` — retrieval and model configuration;
* ``*_opacities.pic`` — gas and CIA opacity subsets;
* ``*_cloudata.pic`` — cloud optical properties for cloudy runs.

These files are used by Tutorial 4 and the cloud post-processing guide.

6. Analyse the posterior
------------------------

Report posterior intervals, inspect chain or nested-sampling diagnostics, draw
posterior predictive spectra, evaluate the P–T and abundance structures, and
plot contribution functions. For cloudy retrievals, compare the total cloud
photosphere with each individual component rather than interpreting the total
curve as a single condensate.
