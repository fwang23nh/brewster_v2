#!/usr/bin/env python

"""Bits for McNuggets: the post-processing tool for Brewster."""
import numpy as np
from scipy.integrate import trapezoid
from collections import namedtuple
from copy import copy
import test_module
import settings
import utils


def teffRM(theta,re_params,sigDist,sigPhot):
    """Append log10(L/Lsun), Teff/K, R/(71492 km), and M/Mjup.

    Integrate the raw diluted spectrum over its loaded wavelength range.
    For MCMC, sigDist is in pc and sigPhot is in magnitudes. Use the same
    distance/calibration draw for L, R and M; Teff is independent of these
    common normalisations. MultiNest already samples physical M and R, so
    these extra uncertainties are not applied a second time.
    """
    args_instance = settings.runargs
    all_params,all_params_values = utils.get_all_parametres(re_params.dictionary)
    params_master = namedtuple('params',all_params)
    params_instance = params_master(*theta)
    if not np.all(np.isfinite([sigDist,sigPhot])) or sigDist < 0 or sigPhot < 0:
        raise ValueError('sigDist and sigPhot must be finite and nonnegative.')
    dist = args_instance.dist
    if not np.isfinite(dist) or dist <= 0:
        raise ValueError('Distance must be finite and positive (pc).')
    D = dist * 3.086e16

    if re_params.samplemode == 'mcmc':
        r2d2 = params_instance.r2d2
        logg = params_instance.logg
    elif re_params.samplemode == 'multinest':
        M = params_instance.M
        R = params_instance.R * 69911e3  # Match test_module.modelspec's input unit.
        if not np.all(np.isfinite([M,R])) or M <= 0 or R <= 0:
            raise ValueError('MultiNest mass and radius must be finite and positive.')
        logg = np.log10(100. * 6.67e-11 * M * 1.898e27 / R**2)
        r2d2 = R**2 / D**2
        # modelspec otherwise draws a different random distance internally.
        args_instance = copy(args_instance)
        args_instance.dist_err = 0.0
    else:
        raise ValueError('Unsupported samplemode: ' + str(re_params.samplemode))
    if not np.all(np.isfinite([r2d2,logg])) or r2d2 <= 0:
        raise ValueError('Invalid radius/distance scaling or gravity.')

    trimspec, photspec,tauspec,cfunc = test_module.modelspec(
        theta,re_params,args_instance,0)
    order = np.argsort(trimspec[0])
    wave = trimspec[0,order]
    flux = trimspec[1,order]
    if (wave.size < 2 or not np.all(np.isfinite([wave,flux]))
            or np.any(wave <= 0) or np.any(np.diff(wave) <= 0)):
        raise ValueError('Spectrum must be finite with distinct positive wavelengths.')
    fbol = trapezoid(flux,wave)
    if not np.isfinite(fbol) or fbol <= 0:
        raise ValueError('Integrated flux must be finite and positive.')
    surface_flux = fbol / r2d2
    t_ff = (surface_flux / 5.670367e-8)**0.25

    if re_params.samplemode == 'mcmc':
        # A Gaussian magnitude error gives a positive multiplicative flux factor.
        calibration = 10.**(-0.4 * np.random.normal(0.,sigPhot))
        D = np.random.normal(dist,sigDist) * 3.086e16
        if not np.isfinite(D) or D <= 0:
            raise ValueError('Distance uncertainty produced a nonpositive distance.')
        r2d2 = r2d2 * calibration
        R = np.sqrt(r2d2) * D
        M = R**2 * (10.**logg / 100.) / (6.67e-11 * 1.898e27)

    l_bol = np.log10(4.*np.pi*R**2 * surface_flux / 3.828e26)
    R = R / 71492e3  # Preserve the post-processing output radius convention.
    result = np.concatenate((theta,np.array([l_bol,t_ff,R,M])),axis=0)
    if not np.all(np.isfinite(result)):
        raise ValueError('Nonfinite post-processing result.')
    return result


def get_endchain(runname,fin):
    """Load final 2000 populated iterations; keep samples and scores aligned."""
    if fin == 1:
        sampler = utils.pickle_load(runname + '.pk1')
        # These emcee compatibility properties are walker-major in saved samplers.
        chain = np.asarray(sampler.chain)
        probs = np.asarray(sampler.lnprobability)
    elif fin == 0:
        chain,probs = utils.pickle_load(runname + '_snapshot.pic')
        chain = np.asarray(chain)
        probs = np.asarray(probs)
    else:
        raise ValueError('fin must be 0 (snapshot) or 1 (finished sampler).')
    if chain.ndim != 3 or probs.shape != chain.shape[:2]:
        raise ValueError('Expected chain[walker,iteration,parameter] and matching scores.')
    if chain.shape[0] == 0 or chain.shape[2] == 0:
        raise ValueError('Empty chain.')
    niter = chain.shape[1]
    if fin == 0:
        # Snapshot storage is padded with all-zero iterations; valid parameters
        # themselves can be zero, so do not count individual nonzero entries.
        populated = np.any(chain != 0.,axis=(0,2))
        filled = np.flatnonzero(populated)
        if filled.size == 0:
            raise ValueError('Snapshot contains no populated iterations.')
        niter = filled[-1] + 1
        if not np.all(populated[:niter]):
            raise ValueError('Snapshot contains an empty iteration before its end.')
    start = max(0,niter-2000)
    ndim = chain.shape[2]
    flatendchain = chain[:,start:niter,:].reshape((-1,ndim))
    flatendprobs = probs[:,start:niter].reshape(-1)
    if (flatendchain.shape[0] == 0 or not np.all(np.isfinite(flatendchain))
            or np.any(np.isnan(flatendprobs)) or np.any(np.isposinf(flatendprobs))
            or not np.any(np.isfinite(flatendprobs))):
        raise ValueError('Empty chain or invalid samples/scores.')
    return flatendchain,flatendprobs,ndim


def getargs(runname):
    """Load the v2 argument object; legacy tuple unpacking is no longer valid."""
    return utils.pickle_load(runname + '_runargs.pic')
