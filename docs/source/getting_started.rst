Getting started
===============

Brewster v2 separates a retrieval into two configuration objects:

``Retrieval_params``
   Defines what is retrieved: sampler, chemistry, gases, vertical profiles,
   P–T profile, clouds, patches, and nuisance parameters.

``ModelConfig``
   Defines model inputs and numerical choices: data, distance, opacity paths,
   pressure and wavelength grids, convolution, and radiative transfer.

Minimal configuration
---------------------

The following cloud-free example shows the main choices. Use the complete files
under ``examples/G570D_emcee`` and ``examples/G570D_nested`` as executable
templates.

.. code-block:: python

   import utils

   gaslist = ["h2o", "co", "co2", "ch4", "nh3", "h2s", "k", "na"]

   re_params = utils.Retrieval_params(
       samplemode="mcmc",
       chemeq=0,
       gaslist=gaslist,
       gastype_list=["U"] * len(gaslist),
       do_fudge=1,
       ptype=1,
       do_clouds=0,
       npatches=1,
       cloud_name=["clear"],
       cloud_type=["None"],
       cloudpatch_index=[[1]],
       particle_dis=["None"],
   )

   model_config = utils.ModelConfig(
       samplemode="mcmc",
       do_fudge=1,
       cloudpath="../Clouds/",
   )

How a model evaluation flows
----------------------------

.. code-block:: text

   parameter vector
        |
        +-- P–T parameters ------> TPmod.set_prof
        +-- gas parameters ------> uniform / non-uniform / equilibrium VMR
        +-- cloud parameters ----> cloud_dic_new.cloud_unpack + atlas
        |
        +-- opacity + radiative transfer ----> model spectrum
        +-- instrument processing -----------> likelihood

Configuration checklist
-----------------------

* Keep ``gaslist`` and ``gastype_list`` in the same order and length.
* Select a ``ptype`` whose parameters and priors are appropriate for the object.
* For every cloud, supply matching entries in ``cloud_name``, ``cloud_type``,
  ``cloudpatch_index``, and ``particle_dis``.
* Pressure is in bar throughout the atmospheric parameterizations.
* Cloud base/top and transition-pressure parameters are sampled as
  :math:`\log_{10}(P/\mathrm{bar})` where their names begin with ``logp``.
* Inspect ``re_params.dictionary`` before starting an expensive retrieval; it
  is the authoritative ordered description of the parameter vector.

Next steps
----------

* :doc:`model_components/temperature_pressure`
* :doc:`model_components/gases`
* :doc:`model_components/clouds`
* :doc:`retrieval_workflow`
