#!/usr/bin/env python

""" Module of bits for non-uniform gas profile"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

__author__ = "Fei Wang"
__copyright__ = "Copyright 2024 - Fei Wang"
__credits__ = ["Fei Wang", "Ben Burningham"]
__license__ = "GPL"
__version__ = "0.2"  
__maintainer__ = ""
__email__ = ""
__status__ = "Development"


def non_uniform_gas(press,logPt,logft,alpha):
    """
    Construct a vertically varying gas volume mixing ratio profile.

    This parameterization defines a power-law decrease in log10(VMR) above a
    transition pressure Pt, and a constant mixing ratio below Pt.

    Parameters
    ----------
    press : ndarray
        Pressure grid (must be positive, increasing).
    logPt : float
        log10 of the transition pressure Pt (same units as `press`).
    logft : float
        log10 of the deep (well-mixed) volume mixing ratio.
    alpha : float
        Power-law slope controlling how rapidly the abundance decreases
        with decreasing pressure above Pt.

    Returns
    -------
    gas_f : ndarray
        log10 of the gas volume mixing ratio profile on the `press` grid.

    Notes
    -----
    The profile is defined as:

        log10(f(P)) = logft + (log10(P) - logPt) / alpha    for P < Pt
        log10(f(P)) = logft                                for P >= Pt

    This ensures continuity at P = Pt.
    """

    gas_f = np.zeros_like(press)
    
    Pt = 10.**logPt
    ft = 10.**logft
    
    for i in range(0,press.size):
        if (press[i] < Pt):
            gas_f[i]=logft+(np.log10(press[i])-logPt) / alpha
        else:
            gas_f[i]=float(logft)
    
    return gas_f




def non_uniform_gas_inverted(press, logPt, logft, alpha):
    """
    Vertically varying gas VMR profile with constant abundance above Pt
    and a gradient below Pt.

    log10(f(P)) = logft                                  for P < Pt
    log10(f(P)) = logft + (log10(P) - logPt) / alpha     for P >= Pt
    """

    gas_f = np.zeros_like(press)

    Pt = 10.0**logPt

    for i in range(press.size):
        if press[i] < Pt:
            gas_f[i] = float(logft)
        else:
            gas_f[i] = logft + (np.log10(press[i]) - logPt) / alpha

    return gas_f


def gas_PT_perturbation(press, gas_profile, logP_ref, dp, abun_var):
    """Apply a smooth, box-like perturbation to a log10 H- profile.

    The perturbation is centred on ``logP_ref``, has a width of ``dp`` dex,
    and changes the log abundance by ``abun_var`` inside that pressure range.
    """
    press = np.asarray(press)
    perturbed = np.array(gas_profile, dtype=float, copy=True)
    log_press = np.log10(press)
    log_p_top = logP_ref - 0.5 * dp
    log_p_base = logP_ref + 0.5 * dp
    mask = (log_press >= log_p_top) & (log_press <= log_p_base)
    perturbed[mask] += abun_var
    return gaussian_filter1d(perturbed, sigma=2.0, mode="nearest")


