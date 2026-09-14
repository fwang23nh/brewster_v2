Atmospheric model components
============================

This section describes the parameterizations currently wired into the v2
retrieval configuration and forward-model path.

.. toctree::
   :maxdepth: 2

   temperature_pressure
   gases
   clouds

Choosing a model
----------------

Model flexibility should follow the information content of the spectrum.
Additional parameters can fit real structure, but can also create degeneracies
between temperature, composition, gravity, radius, and clouds. Compare models
with posterior predictive checks and an appropriate model-comparison metric;
do not select a parameterization only because it gives the smallest residuals.
