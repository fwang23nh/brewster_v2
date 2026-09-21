"""Run from the Brewster repository root to reproduce the documentation figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gas_nonuniform import non_uniform_gas, non_uniform_gas_inverted


press = np.logspace(-4, 2, 301)  # Increasing pressure, in bar.
log_abund = -4.0
p_ref = -1.0  # log10(P_ref / bar): P_ref = 0.1 bar.
alphas = [2.0, -2.0]

fig, axes = plt.subplots(1, 3, figsize=(10, 4.5), sharex=True, sharey=True)
axes[0].plot(np.full_like(press, log_abund), press, color="#333333", lw=2,
             label="Constant abundance")

for ax, profile in zip(axes[1:], [non_uniform_gas, non_uniform_gas_inverted]):
    for alpha, color, style in zip(alphas, ["#0072B2", "#D55E00"], ["-", "--"]):
        log_vmr = profile(press, p_ref, log_abund, alpha)
        ax.plot(log_vmr, press, color=color, ls=style, lw=2,
                label=rf"$\alpha = {alpha:+g}$")
    ax.axhline(10.0**p_ref, color="0.5", ls=":", lw=1)
    ax.text(-6.8, 0.075, r"$P_{\rm ref}=0.1$ bar", fontsize=9)

for ax, title in zip(axes, ["U: uniform", "N: gradient above", "I: gradient below"]):
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_ylim(press[-1], press[0])  # Lower pressures at the top.
    ax.set_xlim(-7, -1)
    ax.set_xticks([-7, -5, -3, -1])
    ax.set_xlabel(r"$\log_{10}$ volume mixing ratio")
    ax.grid(alpha=0.15)
    ax.legend(loc="lower left", fontsize=9)

axes[0].set_ylabel("Pressure / bar")
fig.suptitle(r"Illustrative gas profiles: $\log_{10} f_{\rm ref}=-4$", fontsize=13)
fig.tight_layout()
output = Path(__file__).resolve().parents[1] / "_static" / "gas_profiles.png"
fig.savefig(output, dpi=180)
plt.close(fig)
