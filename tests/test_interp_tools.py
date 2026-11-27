"""interp_tools reproduce the per-layer spline loops they replaced in modelspec."""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

import interp_tools

rng = np.random.default_rng(1)
nT, nP, nlayers, ng = 27, 30, 64, 13
Tgrid = np.sort(rng.uniform(100, 4500, nT))
Pgrid = np.linspace(-5, 2.5, nP)                     # log10 P, as in the CE tables
press = np.logspace(-4.5, 2.8, nlayers)              # reaches past Pgrid at the top/bottom
temp = np.r_[60.0, rng.uniform(80, 4600, nlayers - 2), 5000.0]   # incl. extrapolation


def test_chemeq_cubic_matches_loop():
    invmr = rng.normal(-6, 2, (nT, nlayers, ng))
    ref = np.empty((nlayers, ng))
    for p in range(nlayers):
        for g in range(ng):
            ref[p, g] = IUS(Tgrid, invmr[:, p, g])(temp[p])
    out = interp_tools.cubic_columns(
        Tgrid, invmr.reshape(nT, nlayers * ng), np.repeat(temp, ng)).reshape(nlayers, ng)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_bff_linear_in_T_matches_loop():
    bff_grid = rng.normal(-8, 2, (nT, nlayers, 3))
    ref = np.empty((3, nlayers))
    for gas in range(3):
        for i in range(nlayers):
            ref[gas, i] = IUS(Tgrid, bff_grid[:, i, gas], k=1)(temp[i])
    out = interp_tools.linear_columns(Tgrid, bff_grid[:, :nlayers, :], temp[:, None]).T
    np.testing.assert_allclose(out, ref, rtol=1e-14, atol=1e-14)


def test_bff_linear_in_P_matches_loop():
    ions = 10 ** rng.normal(-8, 2, (nT, nP, 3))
    ref = np.empty((nT, nlayers, 3))
    for gas in range(3):
        for i in range(nT):
            ref[i, :, gas] = IUS(Pgrid, np.log10(ions[i, :, gas]), k=1)(np.log10(press))
    out = interp_tools.linear_columns(
        Pgrid, np.moveaxis(np.log10(ions), 1, 0)[:, :, None, :],
        np.log10(press)[None, :, None])
    np.testing.assert_allclose(out, ref, rtol=1e-14, atol=1e-14)
