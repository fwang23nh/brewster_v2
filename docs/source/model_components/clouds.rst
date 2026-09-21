Cloud parameterizations
=======================

Clouds have three independent choices:

* vertical geometry: ``deck`` or ``slab``;
* wavelength behaviour: ``grey``, ``powerlaw``, or a named Mie opacity;
* for Mie clouds, particle distribution: ``hansen`` or ``log_normal``.

Set these through ``cloud_name``, ``cloud_type``, ``particle_dis``, and
``cloudpatch_index`` in :class:`utils.Retrieval_params`.

Vertical geometry
-----------------

Slab
   A finite cloud between a top and base pressure. Its reference optical depth
   is distributed through the occupied layers with :math:`d\tau/dP \propto P`.
   Parameters include total optical depth at 1 µm, log base pressure, and
   thickness ``dp`` in pressure decades.

Deck
   An optically thick cloud whose reference pressure is retrieved. The optical
   depth per unit pressure grows exponentially with pressure; in the analytic
   prescription, cumulative optical depth from zero pressure is one at the
   reference pressure. This is often called the deck top, but it is not a
   sharp boundary. ``dp`` controls its decay scale.

Visual comparison
~~~~~~~~~~~~~~~~~

.. figure:: /_static/cloud_vertical_profiles.png
   :alt: Slab and deck layer optical depths versus pressure for dp of 0.3 and 1. Slabs have finite boundaries, while decks extend into the deep atmosphere.
   :width: 100%

   Grey cloud profiles returned by :func:`cloud_dic_new.atlas` on 64 layers.
   Pressure increases downward. The horizontal axis is logarithmic above
   0.001 and linear near zero so cloud-free slab layers remain visible.
   These are illustrative profiles, not retrieved cloud properties.

For the **slab**, the base pressure is 1 bar and total reference optical
depth is 3. The dashed line marks the base; coloured dotted lines mark
the two tops. The top pressure is

.. math::

   P_\mathrm{top} = P_\mathrm{base}\,10^{-dp}.

Increasing ``dp`` extends the slab upward while keeping its base fixed.
At fixed total optical depth, it redistributes that optical depth over a
larger pressure interval. The code integrates :math:`d\tau/dP\propto P`
over layer boundaries, including partially occupied top and base layers.
The plotted values are optical depths **per layer**, not cumulative optical
depth or :math:`d\tau/dP`; their sum is the slab's total optical depth.

For the **deck**, the dashed line marks :math:`P_0=1` bar. Both examples
use the same reference pressure, with

.. math::

   S = P_0(1-10^{-dp}), \qquad
   \frac{d\tau}{dP} =
   \frac{\exp[(P-P_0)/S]}{S[1-\exp(-P_0/S)]}.

Here ``dp`` sets the pressure scale :math:`S`, not a finite cloud thickness.
A smaller positive ``dp`` concentrates the rise more tightly around
:math:`P_0`; larger values give a more extended upper tail.
Unlike the slab, the deck has no independently retrieved total optical depth.

The plotted deck includes the implementation's deep-layer rule: when either
layer-boundary exponent exceeds 10, ``atlas`` assigns that layer
:math:`\Delta\tau=100`. The deep plateau therefore reflects a numerical
prescription. On a finite pressure grid, a cumulative sum also omits material
above the grid and need not equal exactly one at a sampled layer centre.

Opacity models
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Model
     - Optical parameters
     - Description
   * - Grey
     - optical depth, single-scattering albedo
     - Wavelength-independent extinction.
   * - Power law
     - optical depth, albedo, ``alpha``
     - Reference optical depth scaled with a wavelength power law.
   * - Mie
     - condensate opacity plus particle-distribution parameters
     - Uses the saved efficiency grids for the condensate named after ``--``
       in the generated cloud identifier.

Wavelength comparison
~~~~~~~~~~~~~~~~~~~~~

.. figure:: /_static/cloud_wavelength_scaling.png
   :alt: Grey extinction is constant with wavelength. Power-law extinction decreases for alpha minus two and increases for alpha plus two; all curves equal one at one micron.
   :width: 85%

   Extinction relative to its value at 1 micron, independent of whether the
   vertical geometry is a slab or deck.

For the generic opacity models, ``clouds_mod.f90`` applies

.. math::

   \Delta\tau_\lambda = \Delta\tau_{1\,\mu\mathrm{m}}
   \left(\frac{\lambda}{1\,\mu\mathrm{m}}\right)^\alpha.

Grey clouds use ``alpha=0``. Negative ``alpha`` gives stronger extinction
at shorter wavelengths, and positive ``alpha`` gives stronger extinction
at longer wavelengths. This ``alpha`` is a wavelength exponent, distinct
from the gas-profile gradient parameter. The single-scattering albedo
``omega`` sets the scattering fraction of extinction; it does not change
the extinction curves shown here.

Mie extinction depends on the condensate's saved efficiency grid and the
particle-size distribution. It cannot be represented by a single universal
curve in this comparison.

Reproduce the figures
~~~~~~~~~~~~~~~~~~~~~

From the Brewster repository root, run
``PYTHONPATH=. python docs/source/_scripts/plot_cloud_profiles.py`` with
NumPy, SciPy, Astropy, and Matplotlib installed. The script uses the actual
``atlas`` profiles and the wavelength scaling implemented in Fortran; it
does not run radiative transfer or load Mie opacity tables.

.. literalinclude:: /_scripts/plot_cloud_profiles.py
   :language: python
   :lines: 3-

Particle distributions
----------------------

``hansen``
   Retrieves Hansen effective-radius parameter ``a`` (stored in log space by
   the configuration) and width ``b``.

``log_normal``
   Retrieves geometric mean radius ``mu`` (stored in log space) and normalized
   width ``sigma``. The Fortran cloud model maps the width onto its internal
   distribution range.

Configuration examples
----------------------

One silicate slab in patch 1:

.. code-block:: python

   do_clouds = 1
   npatches = 1
   cloud_name = ["Mg2SiO4"]
   cloud_type = ["slab"]
   cloudpatch_index = [[1]]
   particle_dis = ["log_normal"]

A patchy atmosphere with an iron deck in both patches and a silicate slab only
in the cloudy patch:

.. code-block:: python

   npatches = 2
   cloud_name = ["Fe", "Mg2SiO4"]
   cloud_type = ["deck", "slab"]
   cloudpatch_index = [[1, 2], [1]]
   particle_dis = ["hansen", "log_normal"]

For generic opacity models, set ``cloud_name`` to ``"grey"`` or
``"powerlaw"``. For a clear atmosphere use ``do_clouds=0`` and
``cloud_name=["clear"]``.

Patch coverage
--------------

With two patches, v2 adds ``fcld`` and uses coverage fractions
``[fcld, 1-fcld]``. ``cloudpatch_index`` contains one-based patch numbers;
the resulting ``ModelConfig.cloudmap`` is a zero/one array with shape
``(npatch, ncloud)``.

Diagnostics
-----------

See :doc:`../postprocessing` for total and species-specific cloud
:math:`\tau=1` pressures and layer-by-layer :math:`\Delta\tau_{1\,\mu m}`.
