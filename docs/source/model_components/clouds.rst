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
   An optically thick cloud whose top is retrieved. The layer optical depth
   grows exponentially below the top; by definition, cumulative cloud optical
   depth is one at the deck-top pressure. ``dp`` controls its decay scale.

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
