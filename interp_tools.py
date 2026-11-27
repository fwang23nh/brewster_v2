"""Vectorised stand-ins for per-layer scipy spline loops in modelspec.

modelspec used to build one ``InterpolatedUnivariateSpline`` per layer (and per
gas) on a shared x grid and evaluate it at one point. These functions fit all
columns at once and evaluate column j at its own point ``xq[j]``. Both give
the same spline as FITPACK (s=0: piecewise linear for k=1, not-a-knot cubic
for k=3), including FITPACK's default extrapolation from the end pieces.
"""

import numpy as np
from scipy.interpolate import make_interp_spline


def linear_columns(x, Y, xq):
    """``InterpolatedUnivariateSpline(x, Y[:, j], k=1)(xq[j])`` for every j.

    x : (n,) increasing; Y : (n, ...); xq : array. Y[0] and xq are broadcast
    together; the result has that broadcast shape.
    """
    x = np.asarray(x, dtype=float)
    Y = np.asarray(Y, dtype=float)
    xq = np.asarray(xq, dtype=float)
    shape = np.broadcast_shapes(Y.shape[1:], xq.shape)
    Y = np.broadcast_to(Y, (x.size,) + shape)
    xq = np.broadcast_to(xq, shape)
    i = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, x.size - 2)
    x0, x1 = x[i], x[i + 1]
    y0 = np.take_along_axis(Y, i[None], axis=0)[0]
    y1 = np.take_along_axis(Y, (i + 1)[None], axis=0)[0]
    t = (xq - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def cubic_columns(x, Y, xq):
    """``InterpolatedUnivariateSpline(x, Y[:, j])(xq[j])`` (k=3) for every j.

    x : (n,) increasing, n >= 4; Y : (n, m); xq : (m,).
    """
    Y = np.asarray(Y, dtype=float)
    xq = np.asarray(xq, dtype=float)
    spl = make_interp_spline(np.asarray(x, dtype=float), Y, k=3, axis=0)
    # B-spline basis at each query point, (m, nbasis); column j uses row j
    D = spl.design_matrix(xq, spl.t, spl.k, extrapolate=True).toarray()
    return np.einsum("jb,bj->j", D, spl.c)
