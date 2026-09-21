Gas abundance profiles and chemistry
====================================

The ``chemeq`` option selects between free abundances and the chemical-
equilibrium table. For free chemistry, each ``gastype_list`` entry selects the
vertical profile assigned to the gas at the same position in ``gaslist``.

Chemistry modes
---------------

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Setting
     - Parameters
     - Behaviour
   * - ``chemeq=0``
     - One or three parameters per gas
     - Free chemistry. Abundances are stored as :math:`\log_{10}` volume
       mixing ratios.
   * - ``chemeq=1``
     - ``mh``, ``co``
     - Interpolates the equilibrium grid in metallicity, C/O, pressure, and
       temperature. Individual gas abundances are not free parameters.

Free-chemistry profiles
-----------------------

``U`` — uniform
   Retrieves ``log_abund`` and applies it at every pressure. This is the most
   compact choice and is appropriate when vertical structure is unconstrained.

``N`` — non-uniform
   Retrieves ``log_abund``, ``p_ref``, and ``alpha``. The forward model calls
   :func:`gas_nonuniform.non_uniform_gas`: abundance follows a log-pressure
   gradient above the reference pressure and is constant below it.

.. math::

   \log_{10} f(P) =
   \begin{cases}
   \log_{10} f_\mathrm{ref} +
   \dfrac{\log_{10}(P/P_\mathrm{ref})}{\alpha},
   & P < P_\mathrm{ref} \\
   \log_{10} f_\mathrm{ref}, & P \ge P_\mathrm{ref}.
   \end{cases}

``I`` — inverted non-uniform
   Retrieves the same three parameters. The forward model calls
   :func:`gas_nonuniform.non_uniform_gas_inverted`: abundance is constant
   above the reference pressure and follows a log-pressure gradient below it.

.. math::

   \log_{10} f(P) =
   \begin{cases}
   \log_{10} f_\mathrm{ref}, & P < P_\mathrm{ref} \\
   \log_{10} f_\mathrm{ref} +
   \dfrac{\log_{10}(P/P_\mathrm{ref})}{\alpha},
   & P \ge P_\mathrm{ref}.
   \end{cases}

For both profiles, ``log_abund`` is :math:`\log_{10} f_\mathrm{ref}` and
``p_ref`` is :math:`\log_{10}(P_\mathrm{ref}/\mathrm{bar})`, despite its name.
Above means lower pressure; below means higher pressure. Both branches meet
at :math:`f_\mathrm{ref}`. The gradient in log abundance versus log pressure
is :math:`1/\alpha`, so positive ``alpha`` means abundance increases with
pressure on the varying branch; negative ``alpha`` reverses that trend.
``alpha=0`` is undefined. ``I`` selects which side varies, rather than
specifying the sign of the gradient.

Visual comparison
-----------------

The three panels below use the same reference volume mixing ratio,
:math:`f_\mathrm{ref}=10^{-4}`. For ``N`` and ``I``, the reference pressure
is :math:`P_\mathrm{ref}=0.1` bar (``p_ref=-1``), with curves for
``alpha=+2`` and ``alpha=-2``. These are illustrative free-chemistry
profiles, not predictions from the equilibrium table or a fitted atmosphere.

.. figure:: /_static/gas_profiles.png
   :alt: Three gas profiles with pressure increasing downward. U is constant; N varies above 0.1 bar; I varies below 0.1 bar. Positive and negative alpha give opposite gradients.
   :width: 100%

   Uniform (``U``), non-uniform (``N``), and inverted non-uniform (``I``)
   profiles evaluated on an increasing pressure grid. Dotted lines mark the
   reference pressure for the two non-uniform profiles.

* ``U`` holds the abundance fixed throughout the atmosphere and has no
  reference-pressure or gradient parameter.
* ``N`` varies at lower pressures and reaches a constant deep abundance.
  Positive ``alpha`` depletes the gas towards the top; negative ``alpha``
  enriches it there.
* ``I`` holds the upper abundance fixed and varies at higher pressures.
  Positive ``alpha`` enriches the gas towards the bottom; negative ``alpha``
  depletes it there.

Changing ``log_abund`` shifts a curve horizontally. Changing ``p_ref`` moves
the transition vertically. Larger :math:`|\alpha|` gives a weaker abundance
gradient because the slope is :math:`1/\alpha`; it is not a transition width.
For example, with ``alpha=2``, increasing pressure by one decade on the
varying branch increases :math:`\log_{10} f` by 0.5 dex.

Reproduce the figure
~~~~~~~~~~~~~~~~~~~~

The script calls the same non-uniform profile functions as the forward
model. Their outputs are already :math:`\log_{10}` VMR, so the horizontal
axis is linear in those returned values. Pressure is plotted logarithmically
and increases downward.

From the Brewster repository root, run
``PYTHONPATH=. python docs/source/_scripts/plot_gas_profiles.py`` with NumPy,
SciPy, and Matplotlib installed. The figure is saved in ``docs/source/_static``.

.. literalinclude:: /_scripts/plot_gas_profiles.py
   :language: python
   :lines: 3-

These profile functions do not enforce the retrieval priors. When choosing
other example parameters, check that the resulting VMRs remain physically
admissible throughout the pressure grid; do not clip an invalid profile to
make it look acceptable. The retrieval applies additional checks in
:mod:`Priors`.

Configuration example
---------------------

.. code-block:: python

   gaslist = ["h2o", "co", "ch4"]
   gastype_list = ["U", "N", "I"]

Water receives one retrieval parameter; carbon monoxide and methane each
receive three, with their gradients on opposite sides of the reference pressure.
Use the same-length lists and verify the result with:

.. code-block:: python

   print(re_params.dictionary["gas"])

Alkalis and H-minus
-------------------

When ``k`` and ``na`` are the final two entries, v2 retrieves a combined alkali
abundance and splits it using the built-in abundance ratio. The analogous
``k``, ``na``, ``cs`` ending uses a combined three-alkali parameter. Bound-free
and free-free H-minus opacity is controlled separately by ``do_bff`` and the
chemical-equilibrium support table loaded by :class:`utils.ModelConfig`.

.. warning::

   The older ``H`` gas-profile branch is present in comments and some prior
   logic, but is not generated by the current ``gas_dic_gen`` interface. Use
   ``U``, ``N``, or ``I`` for new v2 configurations.
