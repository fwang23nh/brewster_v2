Retrieval post-processing
=========================

After a retrieval, post-processing evaluates spectra for posterior samples over
a wider wavelength range and derives distributions of bolometric luminosity,
effective temperature, radius, and mass. The expensive forward-model calls can
run in parallel on a cluster; the resulting array can then be downloaded for
local analysis and plotting.


Driver and helper
-----------------

Use the root-level ``mcnuggets_TEMPLATE.py`` with ``nugbits_TEMPLATE.py``.
The driver restores the v2 retrieval configuration, distributes samples over
MPI ranks, and writes ``<runname>_postprod.pic``. The helper evaluates each
sample and appends ``l_bol``, ``t_ff``, ``R``, and ``M``.

The older ``mcnuggets_general.py`` uses a different helper and output layout;
it is not the driver for this workflow.

Prepare the retrieval and cluster environment
---------------------------------------------

Keep these files together in the directory specified by ``outdir``:

* ``<runname>_runargs.pic``: the saved v2 model arguments, including distance,
  gas list, pressure grid, and opacity paths.
* ``<runname>_configs.pic``: the configuration dictionary containing
  ``re_params``, which defines the retrieval parameter order.
* ``<runname>.pk1`` for ``fin = 1``: a completed sampler with ``chain`` and
  ``lnprobability`` attributes; or ``<runname>_snapshot.pic`` for ``fin = 0``:
  the saved ``(chain, probs)`` pair.
* ``<runname>_cloudata.pic`` when ``re_params.dictionary['cloud']`` is nonempty.

Use the Brewster environment and compiled modules compatible with the saved
retrieval. The cluster also needs ``mpi4py``, a compatible MPI installation,
the molecular opacity files referenced by the saved arguments, and
``data/CIA_DS_aug_2015.dat``. Run from the Brewster repository root so that local
imports and relative data paths resolve.

The v2 driver rebuilds molecular and CIA opacity arrays for the expanded
wavelength interval; it does not simply reuse the retrieval's narrow-band
opacity array. Check ``args_instance.xpath`` and ``args_instance.xlist`` on the
cluster, and ensure all required opacity sources cover the chosen interval.

Configure a run
---------------

Copy the root-level ``mcnuggets_TEMPLATE.py`` to a run-specific driver, for
example ``mcnuggets_my_run.py``, and edit its existing configuration values:

.. code-block:: python

   runname = "my_retrieval"
   outdir = "/path/to/retrieval/results/"  # Keep the trailing slash.
   fin = 0                              # 0: snapshot; 1: completed sampler.
   sigDist = 0.02                        # Distance uncertainty in pc; example only.
   sigPhot = 0.02                        # Flux-calibration uncertainty in magnitudes.
   w1 = 0.7                             # Wavelength limits in micrometres.
   w2 = 20.0
   testrun = 1
   testlen = 10

These are assignments inside the template, not command-line arguments.
Replace the uncertainty values with those appropriate to your observations.
``sigDist`` is an uncertainty on distance, not parallax; ``sigPhot`` is not a
fractional flux uncertainty.

``nugbits_TEMPLATE.get_endchain`` selects the last 2,000 iterations from each
walker. For snapshots, it estimates the populated iteration count from nonzero
chain entries. Inspect the chain and establish burn-in and convergence before
using this selection; it is not a convergence test, and short or unusually
populated chains require particular care. The template processes all selected
rows, or the final ``testlen`` rows when ``testrun = 1``.

.. important::

   This driver loads emcee-style chains. It is not a ready-to-run MultiNest
   post-processor. Although ``teffRM`` contains a MultiNest branch, its later
   mass calculation accesses ``params_instance.logg`` even though that branch
   computes a local ``logg`` from retrieved ``M`` and ``R``. The sample loader,
   gravity handling, and radius convention need checking before adapting this
   workflow to a MultiNest retrieval.

Run on the cluster
------------------

First submit a small test using the same environment and data as the retrieval.
For example, within an allocated MPI job:

.. code-block:: bash

   cd /path/to/brewster_v2
   mpiexec -n 4 python mcnuggets_my_run.py

Use your cluster's scheduler and supported MPI launcher. An illustrative Slurm
submission script is:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=brewster-post
   #SBATCH --nodes=1
   #SBATCH --ntasks=4
   #SBATCH --cpus-per-task=1
   #SBATCH --time=00:30:00
   #SBATCH --output=brewster-post-%j.log

   set -euo pipefail
   cd /path/to/brewster_v2
   # Activate your Brewster environment and load the matching MPI modules here.
   export OMP_NUM_THREADS=1
   srun python mcnuggets_my_run.py

Add your site's account, partition, and memory settings and adjust the time and
resource requests for the opacity grid and number of samples. Submit with
``sbatch postprocess.slurm`` after saving the script under that name.


What is calculated?
-------------------

For each sample, ``teffRM(theta, re_params, sigDist, sigPhot)`` calls the forward
model and processes the spectrum with convolution disabled. It uses the saved
``do_scales`` setting; wavelength shifting follows ``proc_spec``'s default.
The flux density is in W m\ :sup:`−2` µm\ :sup:`−1`, and integration over
wavelength in µm gives an observed flux in W m\ :sup:`−2`.

Writing :math:`q = (R/d)^2`, the derived quantities follow

.. math::

   \begin{aligned}
   F_{\rm bol} &\simeq \sum_j F_{\lambda,j}\,\Delta\lambda_j, \\
   L_{\rm bol} &= 4\pi d^2 F_{\rm bol}, \\
   T_{\rm eff} &= \left(\frac{F_{\rm bol}}{q\sigma_{\rm SB}}\right)^{1/4}, \\
   R &= d\sqrt{q}, \\
   M &= \frac{gR^2}{G}.
   \end{aligned}


The helper uses centred bin widths for interior wavelength samples and omits
the endpoints. The default 0.7–20 µm interval is a finite-band approximation
to bolometric flux. Check that omitted flux is negligible for your atmosphere;
a longer wavelength interval is only useful if the opacity data support it.
Also verify increasing wavelength order and finite, positive integrated flux.

For the MCMC path, luminosity and temperature use the retrieved ``r2d2`` and
nominal distance. The helper adds independent Gaussian distance and
photometric-calibration perturbations when deriving radius, and uses that radius
with retrieved ``logg`` to derive mass. Thus these extra uncertainties are not
propagated into all four quantities. It converts ``logg`` from cgs to SI and
returns radius using 71,492 km per Jupiter radius and mass using
:math:`1.898\times10^{27}` kg per Jupiter mass.

The helper currently replaces NaN spectral fluxes with zero before integration.
A finite output therefore does not establish that the spectrum was valid:
inspect representative spectra and investigate invalid fluxes before reporting
physical results. Negative perturbed radius-squared scaling can also produce
invalid radius and mass values.

Load the output locally
-----------------------

Copy the result and matching configuration file from the cluster. The following
code loads the v2 pickle:

.. code-block:: python

   from pathlib import Path
   import pickle
   import numpy as np
   import utils

   path = Path("/path/to/local/retrieval/results")
   runname = "my_retrieval"

   with (path / f"{runname}_postprod.pic").open("rb") as handle:
       postproddata = np.asarray(pickle.load(handle))

   with (path / f"{runname}_configs.pic").open("rb") as handle:
       configs = pickle.load(handle)
   re_params = configs["re_params"]
   retrieval_names, _ = utils.get_all_parametres(re_params.dictionary)
   all_params = list(retrieval_names) + ["l_bol", "t_ff", "R", "M"]

Each row contains the original retrieval parameters followed by:

.. list-table:: Appended columns in the v2 output
   :header-rows: 1

   * - Column
     - Name
     - Quantity
   * - ``-4``
     - ``l_bol``
     - :math:`\log_{10}(L_{\rm bol}/L_\odot)`
   * - ``-3``
     - ``t_ff``
     - Effective temperature in K
   * - ``-2``
     - ``R``
     - Radius in Jupiter radii (71,492 km convention)
   * - ``-1``
     - ``M``
     - Mass in Jupiter masses

Validate before plotting
------------------------

Recover parameter names from the matching configuration rather than hard-coding
indices or repeatedly appending labels in a notebook cell. Check the full array,
not only whether luminosity equals ``-inf``:

.. code-block:: python

   if postproddata.ndim != 2 or postproddata.shape[1] != len(all_params):
       raise ValueError("Output columns do not match the v2 retrieval configuration.")
   if postproddata.shape[0] == 0:
       raise ValueError("The post-processing array is empty.")

   finite = np.isfinite(postproddata).all(axis=1)
   positive = (postproddata[:, -3:] > 0).all(axis=1)
   valid = finite & positive
   print(f"Valid rows: {valid.sum()} / {len(valid)}")
   if not valid.all():
       bad_rows = np.flatnonzero(~valid)
       raise ValueError(f"Investigate invalid post-processing rows: {bad_rows[:10]}")

A negative ``l_bol`` is normal for luminosities below the Sun's; it is the
logarithm, not a negative luminosity. Diagnose failed rows in the spectrum,
wavelength integration, parameter mapping, or uncertainty perturbations before
plotting. Silently dropping them can change the inferred posterior.

Plot and summarise the posterior
--------------------------------

The notebook uses ``corner`` to display the retrieval parameters together with
the appended quantities. For a compact view of just the derived quantities:

.. code-block:: python

   import corner
   import matplotlib.pyplot as plt

   derived = postproddata[:, -4:]
   derived_labels = [
       r"$\log_{10}(L_{\rm bol}/L_\odot)$",
       r"$T_{\rm eff}$ [K]",
       r"$R/R_{\rm Jup}$",
       r"$M/M_{\rm Jup}$",
   ]
   fig = corner.corner(
       derived,
       labels=derived_labels,
       quantiles=[0.16, 0.5, 0.84],
       show_titles=True,
       plot_datapoints=False,
       scale_hist=False,
       smooth=0.2,
       smooth1d=0.1,
   )
   fig.savefig(path / f"{runname}_derived_corner.png", dpi=200, bbox_inches="tight")
   plt.show()

To reproduce the full posterior plot, use ``postproddata`` instead of
``derived`` and ``labels=all_params``. Summarise the same samples with median and
16th/84th percentiles:

.. code-block:: python

   q16, q50, q84 = np.percentile(derived, [16, 50, 84], axis=0)
   for name, median, lower, upper in zip(
       ["l_bol", "t_ff", "R", "M"], q50, q50 - q16, q84 - q50
   ):
       print(f"{name}: {median:.4g} (-{lower:.4g}, +{upper:.4g})")

These are marginal posterior intervals, conditional on the retrieval model,
wavelength coverage, and uncertainty treatment described above. A successful
cluster job or corner plot alone does not scientifically validate the results.

Related analysis
----------------

See :doc:`tutorials` for retrieval-analysis notebooks,
:doc:`tutorials/Brewster_v2_Tutorial_6_Cloud_Postprocessing` for cloud and
contribution-function plots, and
:doc:`tutorials/Brewster_v2_Tutorial_7_Gas_Abundances_vs_Chemical_Equilibrium`
for abundance-derived C/O and metallicity diagnostics.
