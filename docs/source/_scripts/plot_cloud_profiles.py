"""Reproduce the cloud figures from the Brewster repository root."""

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from cloud_dic_new import atlas


output = Path(__file__).resolve().parents[1] / "_static"
press = np.logspace(-4, 2.4, 64)  # Layer pressures in bar, increasing downward.
fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), sharey=True)

for ax, geometry in zip(axes, ["slab", "deck"]):
    # atlas reads the cloud geometry from the generated identifier.
    config = SimpleNamespace(dictionary={
        "cloud": {"patch 1": {f"grey cloud {geometry}": {}}}
    })
    for dp, color in zip([0.3, 1.0], ["#0072B2", "#D55E00"]):
        # Rows: slab total tau (unused for deck), log pressure, dp, albedo, alpha.
        params = np.array([[3.0], [0.0], [dp], [0.5], [0.0]])
        cloudprof, _, _ = atlas(config, params, press)
        dtau = cloudprof[:, 0]
        assert np.all(np.isfinite(dtau)) and np.all(dtau >= 0)
        if geometry == "slab":
            np.testing.assert_allclose(dtau.sum(), 3.0)
            ax.axhline(10.0**(-dp), color=color, ls=":", lw=1)
        ax.plot(dtau, press, color=color, lw=2, label=f"dp = {dp:g}")
    ax.axhline(1.0, color="0.3", ls="--", lw=1)
    # A linear region at zero retains the empty layers of the slab.
    ax.set_xscale("symlog", linthresh=0.001)
    ax.set_xlim(left=0)
    ax.set_xticks([0, 0.001, 0.1, 10, 1000] if geometry == "deck"
                 else [0, 0.001, 0.01, 0.1, 1])
    ax.set_yscale("log")
    ax.set_ylim(press[-1], press[0])
    ax.set_xlabel(r"Layer optical depth $\Delta\tau_{1\,\mu m}$")
    ax.set_title("Slab: base = 1 bar, total tau = 3" if geometry == "slab"
                 else "Deck: reference pressure = 1 bar")
    ax.grid(alpha=0.15)
    ax.legend(loc="upper right")
axes[0].set_ylabel("Pressure / bar")
fig.tight_layout()
fig.savefig(output / "cloud_vertical_profiles.png", dpi=180)
plt.close(fig)

wavelength = np.logspace(np.log10(0.5), np.log10(15.0), 301)  # Microns.
fig, ax = plt.subplots(figsize=(7, 4))
for alpha, color, style in zip([0, -2, 2], ["#333333", "#0072B2", "#D55E00"],
                              ["-", "--", "-."]):
    # clouds_mod.f90 scales extinction by wavelength (in microns) ** alpha.
    scaling = (wavelength / 1.0)**alpha
    ax.loglog(wavelength, scaling, color=color, ls=style, lw=2,
              label="Grey (alpha = 0)" if alpha == 0 else f"Power law: alpha = {alpha:+g}")
ax.axvline(1.0, color="0.5", ls=":", lw=1)
ax.set_xlabel(r"Wavelength / $\mu$m")
ax.set_ylabel(r"Extinction ratio $\Delta\tau_\lambda / \Delta\tau_{1\,\mu m}$")
ax.set_title("Grey and power-law cloud extinction")
ax.grid(alpha=0.15)
ax.legend()
fig.tight_layout()
fig.savefig(output / "cloud_wavelength_scaling.png", dpi=180)
plt.close(fig)
