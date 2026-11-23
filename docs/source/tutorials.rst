.. Brewster_v2 documentation master file, created by
   sphinx-quickstart on Tue Jan 27 13:08:34 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Tutorials
=========

The notebooks use a common structure: purpose, prerequisites, expected outputs,
interpretation notes, worked sections, and checks and next steps. Each notebook
has its own example inputs; the sequence does not assume that one notebook's
variables or saved results remain in memory for the next.

Start with Tutorials 1–3 for configuration, spectrum generation, and instrument
response files. Tutorials 4 and 4.5 analyse clear and cloudy MCMC retrievals.
Tutorials 5–9 cover specialized priors, cloud diagnostics, chemistry, temperature
profiles, and opacity tables.

Before running a notebook
-------------------------

Use a fresh kernel and run cells in order. Read the prerequisites and edit
local paths before loading data or launching sampling. The documentation build
does not execute notebooks: displayed outputs can come from earlier runs and
are not evidence that the current code or configuration has been validated.
Tutorial 4.5's outputs have been cleared following its likelihood/BIC correction
and must be regenerated from matching retrieval products.

For a complete first run with explicit command-line instructions, start with
:doc:`quickstart` and :doc:`first_retrieval`. Refer to
:doc:`observations_instruments`, :doc:`priors_likelihood`, and
:doc:`running_restarting_monitoring` when adapting notebook settings.

Worked notebooks
-----------------

.. toctree::
   :maxdepth: 1

   Tutorial 1: Configure a retrieval <tutorials/Brewster_v2_Tutorial_1_ClassIntro.ipynb>
   Tutorial 2: Generate a spectrum <tutorials/Brewster_v2_Tutorial_2_Make_a_Spectrum.ipynb>
   Tutorial 3: Prepare instrument response files <tutorials/Brewster_v2_Tutorial_3_R_file_template.ipynb>
   Tutorial 4: Analyse a clear-atmosphere retrieval <tutorials/Brewster_v2_Tutorial_4_Clear_Retrieval_Analysis.ipynb>
   Tutorial 4.5: Analyse a cloudy retrieval <tutorials/Brewster_v2_Tutorial_4.5_Cloudy_Retrieval_Analysis.ipynb>
   Tutorial 5: Inspect MultiNest priors <tutorials/Brewster_v2_Tutorial_5_multinest_prior_intro.ipynb>
   Tutorial 6: Inspect cloud diagnostics <tutorials/Brewster_v2_Tutorial_6_Cloud_Postprocessing.ipynb>
   Tutorial 7: Compare retrieved and equilibrium chemistry <tutorials/Brewster_v2_Tutorial_7_Gas_Abundances_vs_Chemical_Equilibrium.ipynb>
   Tutorial 8: Configure a gradient P–T profile <tutorials/Brewster_v2_Tutorial_8_TP_Profile_Type_4.ipynb>
   Tutorial 9: Inspect gas opacity tables <tutorials/Brewster_v2_Tutorial_9_Gas_Opacity.ipynb>

For posterior distributions of bolometric luminosity, effective temperature,
radius, and mass, see the separate :doc:`postprocessing` guide.
