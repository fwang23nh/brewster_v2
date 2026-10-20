Pressure–temperature parameterizations
======================================

Set the parameterization with ``ptype`` in :class:`utils.Retrieval_params`.
Profiles are evaluated by :func:`TPmod.set_prof` on ``ArgsGen.press``.
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
     - ``Tbottom``, ``dTdP1`` … ``dTdP6``
     - Retrieves logarithmic temperature gradients at six pressures from
       the model bottom to its top, evenly spaced in log pressure, and integrates upward.
   * - ``7``
     - Mollière hybrid radiative–convective
     - ``Tint``, ``alpha``, ``lndelta``, ``T1``, ``T2``, ``T3``
     - Joins upper spline temperatures to an Eddington radiative structure and
       a deep dry adiabat.
   * - ``9``
     - Fixed input profile
     - none
     - Interpolates ``ModelConfig.prof`` and does not retrieve temperatures.

Type 4 conventions and priors
-----------------------------

``dTdP1`` is the deepest gradient and the last ``dTdP`` parameter is the
highest. These dimensionless parameters represent :math:`d\ln T/d\ln P` at
logarithmically spaced knots from ``press[-1]`` to ``press[0]``.
These are the bounds of ``10**logfinePress`` generated in ``ArgsGen``; all
knots lie within the model grid, including both endpoints. Quadratic interpolation
provides gradients on the fine grid. Only the length of ``coarsePress`` is
used for these knots. The default is six knots and 100 fine layers, including
when only ``num_finePress`` is supplied. Custom knot counts require user-defined
gradient distributions and priors, with at least three knots for quadratic
interpolation.

``Tbottom`` anchors the unsmoothed profile at ``press[-1]``. The default fine
grid extends from :math:`10^{-4}` to :math:`10^{2.4}` bar, so this anchor is
approximately 251 bar and coincides with the deepest gradient knot.
For six knots, bottom-to-top log10 pressures in bar are 2.4, 1.12, -0.16,
-1.44, -2.72, and -4. Integration covers the full model grid using the
gradient at the deeper endpoint of each interval; there is no fixed
0.001-bar isothermal cutoff. A Gaussian kernel with a standard deviation of five layers then
smooths the profile, including its endpoints: the returned bottom temperature
can differ from ``Tbottom``. Results therefore depend on fine-grid resolution.

The default MCMC gradient priors are Gaussians truncated at three standard
deviations on each side. For ``MC_init_dis=['truncated_gaussian', mu, sigma]``,
``MC_prior_range=[3, 3]`` specifies sigma counts, not physical bounds.
``MC_P0_gen`` draws normal proposals and retains walkers satisfying these
bounds and the joint prior. The corresponding MultiNest specifications are
``['truncated_gaussian', mu, sigma, 3, 3]``. ``Tbottom`` has a uniform prior
from 2000 to 10000 K in both modes. Custom knot counts have no default gradient
priors; set both MCMC specifications and ``Multinest_prior`` when using both
samplers.

Both samplers retain the existing strict requirement that
the smoothed profile be below 6000 K. All profiles are floored at 100 K, making
the lower check of 1 K redundant. Type 4 has no 4000 K cap, so the upper check
can reject profiles. MultiNest applies this coupled constraint in the likelihood
before evaluating a spectrum. These checks allow inversions and do not impose an
additional monotonicity constraint.

Earlier versions used knots from 1000 to 0.001 bar and an isothermal region
above 0.001 bar. Existing parameter vectors and pressure-dependent priors
therefore have a different interpretation with the model-bound knots;
reversing gradient order alone does not preserve those profiles. Use the
original implementation to reproduce old retrievals, and refit or justify
reference-gradient priors on the new knot pressures. Tutorial 8 derives both
fitting and plotting knots from the model grid. Reference-profile coverage
must still be checked separately from model-grid coverage.

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
