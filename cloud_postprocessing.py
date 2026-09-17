"""Cloud-species diagnostic helpers for Brewster v2.

Brewster's forward model reports the pressure where the *sum* of the cloud
extinction reaches optical depth one.  This module obtains the equivalent
curve for every cloud component by making diagnostic forward-model calls with
one column of ``args_instance.cloudmap`` enabled at a time.  Consequently the
calculation uses exactly the same v2 cloud opacities and Fortran routines as
the retrieval itself.
"""

from __future__ import annotations

from copy import copy
from collections import namedtuple
from dataclasses import dataclass
import math

import numpy as np

import test_module
import cloud_dic_new


@dataclass(frozen=True)
class CloudPhotospheres:
    """Pressure curves returned by :func:`get_cloud_photospheres`.

    All wavelength-dependent arrays use ascending wavelength order, matching
    the spectrum returned by ``test_module.modelspec``.
    """

    wavelength: np.ndarray
    pressure: np.ndarray
    contribution: np.ndarray
    total_cloud: np.ndarray
    gas: np.ndarray
    species: np.ndarray
    labels: tuple[str, ...]


@dataclass(frozen=True)
class CloudLayerOpticalDepth:
    """Per-layer cloud optical depth at Brewster's 1 micron reference."""

    pressure: np.ndarray
    optical_depth: np.ndarray
    cumulative_optical_depth: np.ndarray
    labels: tuple[str, ...]


@dataclass(frozen=True)
class CloudSpeciesMasses:
    """Condensed species mass and formula-unit columns by cloud component."""

    pressure: np.ndarray
    mass: np.ndarray
    formula_units: np.ndarray
    labels: tuple[str, ...]
    species_keys: tuple[str, ...]
    stoichiometry: tuple[dict[str, float], ...]

    @property
    def column_mass(self) -> np.ndarray:
        """Column mass in g cm^-2, shape ``(npatch, ncloud)``."""

        return np.nansum(self.mass, axis=2)

    @property
    def column_formula_units(self) -> np.ndarray:
        """Formula-unit column density in cm^-2, shape ``(npatch, ncloud)``."""

        return np.nansum(self.formula_units, axis=2)


AVOGADRO = 6.02214076e23
PI = math.pi

ATOMIC_MASS = {
    "Mg": 24.305,
    "Si": 28.0855,
    "O": 15.999,
    "Fe": 55.845,
}


def _molar_mass(formula: dict[str, float]) -> float:
    return sum(ATOMIC_MASS[element] * count for element, count in formula.items())


SPECIES_PROPERTIES = {
    "sio": {
        "rho": 2.2,
        "molar_mass": _molar_mass({"Si": 1.0, "O": 1.0}),
        "stoichiometry": {"Si": 1.0, "O": 1.0},
    },
    "mg2sio4": {
        "rho": 3.27,
        "molar_mass": _molar_mass({"Mg": 2.0, "Si": 1.0, "O": 4.0}),
        "stoichiometry": {"Mg": 2.0, "Si": 1.0, "O": 4.0},
    },
    "fe": {
        "rho": 7.87,
        "molar_mass": _molar_mass({"Fe": 1.0}),
        "stoichiometry": {"Fe": 1.0},
    },
    "mgsio3": {
        "rho": 3.66,
        "molar_mass": _molar_mass({"Mg": 1.0, "Si": 1.0, "O": 3.0}),
        "stoichiometry": {"Mg": 1.0, "Si": 1.0, "O": 3.0},
    },
    "sio2": {
        "rho": 2.65,
        "molar_mass": _molar_mass({"Si": 1.0, "O": 2.0}),
        "stoichiometry": {"Si": 1.0, "O": 2.0},
    },
}


def _species_key(label: str) -> str:
    clean = str(label).strip().lower()
    clean = clean.removesuffix(".mieff").removesuffix(".dhs")
    if "mg2sio4" in clean or "forsterite" in clean:
        return "mg2sio4"
    if "mgsio3" in clean or "enstatite" in clean:
        return "mgsio3"
    if "sio2" in clean:
        return "sio2"
    if "sio" in clean:
        return "sio"
    if clean == "fe" or "iron" in clean:
        return "fe"
    return clean


def _radius_bin_widths(radius):
    """Return the EGP radius-bin widths used by Brewster's cloud routines."""

    radius = np.asarray(radius, dtype=float)
    if radius.ndim != 1 or radius.size == 0:
        raise ValueError("args_instance.mierad must be a non-empty 1D array")

    vrat = 2.2
    pw = 1.0 / 3.0
    f2 = (2.0 / (1.0 + vrat)) ** pw * vrat ** (pw - 1.0)
    return f2 * radius


def _material_columns_from_weights(weights, radius, species):
    props = SPECIES_PROPERTIES[species]
    particle_volume = (4.0 / 3.0) * PI * radius**3
    mass_per_particle = props["rho"] * particle_volume
    formula_units_per_particle = (
        props["rho"] * AVOGADRO / props["molar_mass"]
    ) * particle_volume
    mass = np.sum(weights * mass_per_particle)
    formula_units = np.sum(weights * formula_units_per_particle)
    return mass, formula_units


def _lognormal_material_column(dtau1, rg_um, sigma_param, radius, dr, qext_1um, species):
    rsig = 1.0 + (sigma_param * 4.0)
    rg = rg_um * 1e-4
    if dtau1 <= 1e-6 or rsig <= 1.0 or rg <= 0.0:
        return 0.0, 0.0

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        arg1 = dr / (np.sqrt(2.0 * PI) * radius * np.log(rsig))
        arg2 = -(np.log(radius / rg)) ** 2 / (2.0 * np.log(rsig) ** 2)
        cross_section = PI * radius**2 * qext_1um
        norm = np.sum(cross_section * arg1 * np.exp(arg2))

    if not np.isfinite(norm) or norm <= 0.0:
        return 0.0, 0.0

    ndz = dtau1 / norm
    weights = ndz * arg1 * np.exp(arg2)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    return _material_columns_from_weights(weights, radius, species)


def _hansen_material_column(dtau1, a_um, b, radius, dr, qext_1um, species):
    a = a_um * 1e-4
    if dtau1 <= 1e-6 or a <= 0.0 or b <= 0.0:
        return 0.0, 0.0

    valid = (radius > 0.0) & (dr > 0.0) & (qext_1um > 0.0)
    if not np.any(valid):
        return 0.0, 0.0

    rv = radius[valid]
    drv = dr[valid]
    qv = qext_1um[valid]

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        arg1 = (-rv / (a * b)) + np.log(drv)
        arg2 = ((1.0 - 3.0 * b) / b) * np.log(rv)
        argext = np.log(qv * PI * rv**2)
        bot = np.sum(np.exp(arg1 + arg2 + argext))

    if not np.isfinite(bot) or bot <= 0.0:
        return 0.0, 0.0

    logcon = math.log(dtau1 / bot)
    gamma_arg = (1.0 - (2.0 * b)) / b
    try:
        log_gamma = math.lgamma(gamma_arg)
    except ValueError:
        return 0.0, 0.0
    ndz = math.exp(
        logcon
        + log_gamma
        - ((((2.0 * b) - 1.0) / b) * math.log(a * b))
    )
    logcon = ((((2.0 * b) - 1.0) / b) * math.log(a * b)) + math.log(ndz) - log_gamma

    weights = np.zeros_like(radius, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        weights[valid] = np.exp(logcon + arg1 + arg2)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    return _material_columns_from_weights(weights, radius, species)


def get_cloud_species_masses(theta, re_params, args_instance) -> CloudSpeciesMasses:
    """Calculate per-layer condensed species masses for each cloud component.

    The calculation mirrors the cloud optical-depth normalization in
    ``clouds_mod.f90``: each layer's retrieved ``dtau1`` at 1 micron fixes the
    particle column distribution, which is then converted to a material mass
    column using species density and molar mass.
    """

    cloudmap = np.asarray(args_instance.cloudmap)
    if cloudmap.ndim != 2:
        raise ValueError("args_instance.cloudmap must have shape (npatch, ncloud)")

    parameter_names, _ = test_module.utils.get_all_parametres(re_params.dictionary)
    if len(parameter_names) != len(theta):
        raise ValueError("theta length does not match the retrieval parameter dictionary")
    parameters = namedtuple("params", parameter_names)(*theta)
    cloudparams = cloud_dic_new.cloud_unpack(re_params, parameters)
    cloudprof, cloudrad, cloudsig = cloud_dic_new.atlas(
        re_params, cloudparams, args_instance.press
    )

    base_tau = np.asarray(cloudprof, dtype=float)
    if base_tau.shape != (np.asarray(args_instance.press).size, cloudmap.shape[1]):
        raise ValueError(
            "cloud profile shape does not match the pressure grid and cloud map"
        )

    radius = np.asarray(args_instance.mierad, dtype=float)
    dr = _radius_bin_widths(radius)
    miewave = np.asarray(args_instance.miewave, dtype=float)
    if miewave.ndim != 1 or miewave.size == 0:
        raise ValueError("args_instance.miewave must be a non-empty 1D array")
    loc1 = int(np.argmin(np.abs(miewave - 1e-4)))

    labels = _unique_labels(np.atleast_1d(args_instance.cloud_opaname), cloudmap.shape[1])
    cloudsize = np.asarray(args_instance.cloudsize, dtype=int)

    npatches, nclouds = cloudmap.shape
    nlayers = base_tau.shape[0]
    mass = np.zeros((npatches, nclouds, nlayers), dtype=float)
    formula_units = np.zeros_like(mass)
    species_keys = []
    stoichiometry = []

    for cloud_index, label in enumerate(labels):
        species = _species_key(label)
        species_keys.append(species)
        props = SPECIES_PROPERTIES.get(species)
        stoichiometry.append({} if props is None else dict(props["stoichiometry"]))
        if props is None:
            continue

        qext_1um = np.asarray(args_instance.cloudata[cloud_index, 1, loc1, :], dtype=float)
        size_code = cloudsize[cloud_index] if cloud_index < cloudsize.size else 0

        for layer_index in range(nlayers):
            dtau1 = float(base_tau[layer_index, cloud_index])
            if size_code == 1:
                layer_mass, layer_units = _hansen_material_column(
                    dtau1,
                    float(cloudrad[layer_index, cloud_index]),
                    float(cloudsig[layer_index, cloud_index]),
                    radius,
                    dr,
                    qext_1um,
                    species,
                )
            elif size_code == 2:
                layer_mass, layer_units = _lognormal_material_column(
                    dtau1,
                    float(cloudrad[layer_index, cloud_index]),
                    float(cloudsig[layer_index, cloud_index]),
                    radius,
                    dr,
                    qext_1um,
                    species,
                )
            else:
                layer_mass, layer_units = 0.0, 0.0

            active_patches = cloudmap[:, cloud_index] != 0
            mass[active_patches, cloud_index, layer_index] = layer_mass
            formula_units[active_patches, cloud_index, layer_index] = layer_units

    return CloudSpeciesMasses(
        pressure=np.asarray(args_instance.press).copy(),
        mass=mass,
        formula_units=formula_units,
        labels=labels,
        species_keys=tuple(species_keys),
        stoichiometry=tuple(stoichiometry),
    )


def get_cloud_mass(theta, re_params, args_instance):
    """Return per-layer condensed cloud mass columns in g cm^-2.

    The returned array has shape ``(npatch, ncloud, nlayers)``.
    """

    return get_cloud_species_masses(theta, re_params, args_instance).mass


def get_cloud_numdens(theta, re_params, args_instance):
    """Return per-layer condensed formula-unit columns in cm^-2.

    This matches the historical ``cloudnumdens`` diagnostic: it is the
    condensate molecule/formula-unit column per layer, not a gas number density
    per cm^3. The returned array has shape ``(npatch, ncloud, nlayers)``.
    """

    return get_cloud_species_masses(theta, re_params, args_instance).formula_units


def _unique_labels(names: np.ndarray, nclouds: int) -> tuple[str, ...]:
    raw = [str(name).strip() or f"cloud {i + 1}" for i, name in enumerate(names[:nclouds])]
    counts: dict[str, int] = {}
    labels = []
    for name in raw:
        counts[name] = counts.get(name, 0) + 1
        labels.append(name if raw.count(name) == 1 else f"{name} ({counts[name]})")
    return tuple(labels)


def _diagnostic_model(theta, re_params, args_instance, cloudmap):
    """Run diagnostics with a temporary cloud map without mutating runargs."""

    isolated_args = copy(args_instance)
    isolated_args.cloudmap = np.asfortranarray(cloudmap, dtype=np.int32)
    return test_module.modelspec(theta, re_params, isolated_args, gnostics=1)


def get_cloud_photospheres(theta, re_params, args_instance) -> CloudPhotospheres:
    """Calculate total, gas, and individual-cloud tau=1 pressure curves.

    Parameters
    ----------
    theta
        A posterior parameter vector, such as the maximum-likelihood sample.
    re_params
        The v2 retrieval-parameter object saved in ``*_configs.pic``.
    args_instance
        The v2 run arguments saved in ``*_runargs.pic``.  ``settings.linelist``,
        ``settings.cia``, and ``settings.cloudata`` must first be restored in
        the same way as for ``test_module.modelspec``.

    Returns
    -------
    CloudPhotospheres
        ``species`` has shape ``(npatch, ncloud, nwave)``.  A value of zero
        means that the component never reaches optical depth one within the
        model pressure grid.  Each cloud-map column is kept separate, even if
        two columns use the same condensate.

    Notes
    -----
    This makes ``ncloud + 1`` forward-model calls.  The first produces the
    total cloud, gas, and contribution diagnostics; the remaining calls
    isolate one cloud component each.  The original ``args_instance`` is not
    modified.
    """

    cloudmap = np.asarray(args_instance.cloudmap)
    if cloudmap.ndim != 2:
        raise ValueError("args_instance.cloudmap must have shape (npatch, ncloud)")

    npatches, nclouds = cloudmap.shape
    if nclouds == 0:
        raise ValueError("the retrieval contains no cloud components")

    spectrum, total_cloud, gas, contribution = _diagnostic_model(
        theta, re_params, args_instance, cloudmap
    )
    wavelength = np.asarray(spectrum[0]).copy()

    # Fortran diagnostics follow inwavenum (descending wavelength), whereas
    # modelspec reverses only the returned spectrum into ascending wavelength.
    total_cloud = np.asarray(total_cloud)[:, ::-1].copy()
    gas = np.asarray(gas)[:, ::-1].copy()
    contribution = np.asarray(contribution)[:, ::-1, :].copy()

    species = np.zeros((npatches, nclouds, wavelength.size), dtype=float)
    for cloud_index in range(nclouds):
        isolated_map = np.zeros_like(cloudmap)
        isolated_map[:, cloud_index] = cloudmap[:, cloud_index]
        _, isolated_cloud, _, _ = _diagnostic_model(
            theta, re_params, args_instance, isolated_map
        )
        species[:, cloud_index, :] = np.asarray(isolated_cloud)[:, ::-1]

    names = np.atleast_1d(args_instance.cloud_opaname)
    labels = _unique_labels(names, nclouds)
    return CloudPhotospheres(
        wavelength=wavelength,
        pressure=np.asarray(args_instance.press).copy(),
        contribution=contribution,
        total_cloud=total_cloud,
        gas=gas,
        species=species,
        labels=labels,
    )


def get_cloud_layer_optical_depth(theta, re_params, args_instance):
    """Return each cloud component's optical thickness in every layer.

    The returned optical depth is defined at 1 micron, which is the reference
    wavelength used by Brewster's cloud-profile parameterization.  Its shape
    is ``(npatch, ncloud, nlayers)``.  ``cumulative_optical_depth`` is summed
    downward from the top of the atmosphere.
    """

    cloudmap = np.asarray(args_instance.cloudmap)
    if cloudmap.ndim != 2:
        raise ValueError("args_instance.cloudmap must have shape (npatch, ncloud)")

    parameter_names, _ = test_module.utils.get_all_parametres(re_params.dictionary)
    if len(parameter_names) != len(theta):
        raise ValueError("theta length does not match the retrieval parameter dictionary")
    parameters = namedtuple("params", parameter_names)(*theta)
    cloudparams = cloud_dic_new.cloud_unpack(re_params, parameters)
    cloudprof, _, _ = cloud_dic_new.atlas(re_params, cloudparams, args_instance.press)

    # atlas constructs the shared cloud profiles; cloudmap determines which
    # components are present in each patch.
    base_tau = np.asarray(cloudprof, dtype=float)
    if base_tau.shape != (np.asarray(args_instance.press).size, cloudmap.shape[1]):
        raise ValueError(
            "cloud profile shape does not match the pressure grid and cloud map"
        )
    # v2 stores one (layer, cloud) profile and uses cloudmap to place those
    # profiles into patches.
    layer_tau = np.broadcast_to(
        base_tau.T[None, :, :],
        (cloudmap.shape[0], cloudmap.shape[1], base_tau.shape[0]),
    ).copy()
    layer_tau *= cloudmap[:, :, None] != 0

    labels = _unique_labels(np.atleast_1d(args_instance.cloud_opaname), cloudmap.shape[1])
    return CloudLayerOpticalDepth(
        pressure=np.asarray(args_instance.press).copy(),
        optical_depth=layer_tau,
        cumulative_optical_depth=np.cumsum(layer_tau, axis=2),
        labels=labels,
    )


def plot_cloud_layer_optical_depth(
    layer_diagnostics,
    patch=0,
    cumulative=False,
    ax=None,
    include_total=True,
):
    """Plot individual-cloud optical thickness against pressure.

    Parameters
    ----------
    layer_diagnostics : CloudLayerOpticalDepth
        Output from :func:`get_cloud_layer_optical_depth`.
    patch : int, optional
        Zero-based patch index.
    cumulative : bool, optional
        Plot optical depth accumulated from the atmosphere top instead of the
        differential optical thickness in each layer.
    ax : matplotlib.axes.Axes, optional
        Existing axes.  A new figure and axes are made when omitted.
    include_total : bool, optional
        Also plot the sum of all cloud components in the selected patch.

    Returns
    -------
    matplotlib.axes.Axes
        The populated axes.
    """

    import matplotlib.pyplot as plt

    values = (
        layer_diagnostics.cumulative_optical_depth
        if cumulative
        else layer_diagnostics.optical_depth
    )
    if not 0 <= patch < values.shape[0]:
        raise IndexError(f"patch must be between 0 and {values.shape[0] - 1}")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), dpi=120)

    pressure = layer_diagnostics.pressure
    for cloud_index, label in enumerate(layer_diagnostics.labels):
        curve = values[patch, cloud_index]
        if np.any(curve > 0):
            ax.plot(curve, pressure, label=label)

    if include_total:
        ax.plot(values[patch].sum(axis=0), pressure, color="black", lw=2,
                label="all clouds")

    if np.any(values[patch] > 0):
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlabel(
        r"Cumulative cloud $\tau_{1\,\mu\mathrm{m}}$"
        if cumulative
        else r"Layer cloud $\Delta\tau_{1\,\mu\mathrm{m}}$"
    )
    ax.set_ylabel("Pressure / bar")
    ax.legend()
    return ax


def tau_label(label):
    """Return a LaTeX tau=1 label from a cloud opacity filename/name."""

    name = str(label).strip().removesuffix(".mieff").removesuffix(".dhs")
    pieces = []
    text = []
    digits = []

    def flush_text():
        if text:
            pieces.append(rf"\mathrm{{{''.join(text)}}}")
            text.clear()

    def flush_digits():
        if digits:
            pieces.append(rf"_{{{''.join(digits)}}}")
            digits.clear()

    for char in name:
        if char.isdigit():
            flush_text()
            digits.append(char)
        elif char.isalpha():
            flush_digits()
            text.append(char)
        elif char == "_":
            flush_digits()
            text.append(r"\_")
        else:
            flush_digits()
            flush_text()
            pieces.append(char)

    flush_digits()
    flush_text()
    return rf"$\tau_{{{''.join(pieces)}}} = 1.0$"


def plot_tau1_segments(
    ax,
    wave_um,
    tau1_press,
    color,
    label,
    ls="--",
    lw=2,
    gauss=None,
    gap_factor=8.0,
    min_points=5,
):
    """Plot a tau=1 curve in continuous wavelength segments.

    This avoids drawing straight lines across wavelength gaps, matching the
    plotting style used in the older GJ499C notebook.
    """

    tau1_press = np.asarray(tau1_press)
    wave_um = np.asarray(wave_um)

    mask = np.isfinite(tau1_press) & (tau1_press > 0) & np.isfinite(wave_um)
    if np.count_nonzero(mask) < min_points:
        return []

    wave = wave_um[mask]
    pressure = tau1_press[mask]

    if gauss is not None:
        from astropy.convolution import convolve

        pressure = convolve(pressure, gauss, boundary="extend")

    dwave = np.diff(wave)
    positive_dwave = dwave[np.isfinite(dwave) & (dwave > 0)]
    median_dwave = np.median(positive_dwave) if positive_dwave.size else None

    if median_dwave is None or not np.isfinite(median_dwave) or median_dwave == 0:
        line, = ax.plot(wave, pressure, color=color, lw=lw, ls=ls, label=label)
        return [line]

    breaks = np.where(dwave > gap_factor * median_dwave)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, len(wave)]

    lines = []
    first = True
    for start, end in zip(starts, ends):
        if end - start < 3:
            continue
        line, = ax.plot(
            wave[start:end],
            pressure[start:end],
            color=color,
            lw=lw,
            ls=ls,
            label=(label if first else "_nolegend_"),
        )
        lines.append(line)
        first = False

    return lines


def normalized_contribution(contribution):
    """Normalize a ``(nwave, nlayers)`` contribution function by wavelength."""

    values = np.asarray(contribution, dtype=float)
    if values.ndim != 2:
        raise ValueError("contribution must have shape (nwave, nlayers)")
    denominator = values.sum(axis=1, keepdims=True)
    return np.divide(values, denominator, out=np.zeros_like(values), where=denominator != 0)
