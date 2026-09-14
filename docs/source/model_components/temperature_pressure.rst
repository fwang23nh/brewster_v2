Pressure–temperature parameterizations
======================================

Set the parameterization with ``ptype`` in :class:`utils.Retrieval_params`.
Profiles are evaluated by :func:`TPmod.set_prof` on ``ModelConfig.press``.
Most analytic profiles are smoothed with a five-layer Gaussian kernel and the
current implementation limits temperatures to the opacity-table range.

Available profiles
------------------

.. list-table::
   :header-rows: 1
   :widths: 8 24 30 38

   * - ``ptype``
     - Model
     - Retrieved parameters
     - When to use it
   * - ``1``
     - Free spline (Line-style)
     - ``gamma``, ``T_1`` … ``T_13``
     - Flexible profile on 13 coarse pressure knots. ``gamma`` controls the
       smoothing prior used during retrieval.
   * - ``2``
     - Madhusudhan & Seager, no inversion
     - ``alpha1``, ``alpha2``, ``logP1``, ``logP3``, ``T3``
     - Compact three-zone monotonic profile with an isothermal deep region.
   * - ``3``
     - Madhusudhan & Seager, inversion
     - type 2 plus ``logP2``
     - Allows the middle atmospheric zone to contain a thermal inversion.
   * - ``4``
     - Zhang-style gradient profile
     - ``Tbottom``, ``dtdp1`` … ``dtdp6``
     - Retrieves logarithmic temperature gradients at six pressures from
       :math:`10^{-3}` to :math:`10^3` bar and integrates upward.
   * - ``7``
     - Mollière hybrid radiative–convective
     - ``Tint``, ``alpha``, ``lndelta``, ``T1``, ``T2``, ``T3``
     - Joins upper spline temperatures to an Eddington radiative structure and
       a deep dry adiabat.
   * - ``77``
     - Mollière hybrid with smoothing prior
     - ``gamma`` plus the type-7 parameters
     - Same physical profile as type 7, with the smoothing-prior machinery.
   * - ``9``
     - Fixed input profile
     - none
     - Interpolates ``ModelConfig.prof`` and does not retrieve temperatures.

Example comparison
------------------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   import TPmod

   press = np.logspace(-4, 2.4, 64)

   # Type 2: alpha1, alpha2, logP1, logP3, T3
   pars = np.array([0.35, 0.15, -1.0, 1.5, 1800.0])
   temperature = TPmod.set_prof(2, None, press, pars)

   plt.plot(temperature, press)
   plt.yscale("log")
   plt.gca().invert_yaxis()
   plt.xlabel("Temperature / K")
   plt.ylabel("Pressure / bar")

Important constraints
---------------------

The priors in :mod:`Priors` enforce parameter ordering and physical bounds that
are not obvious from the raw parameter names. In particular, transition
pressures must remain ordered, type-7 upper temperatures must connect sensibly
to the radiative profile, and evaluated temperatures must remain within the
model opacity range. Treat ``utils.Retrieval_params.pt_dic_gen`` as the source
of default parameter definitions and :mod:`Priors` as the source of coupled
constraints.
