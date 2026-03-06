#!/usr/bin/env python


""" Bits for McNuggets: the post-processing tool for brewster"""
from __future__ import print_function

from builtins import str
from builtins import range
import numpy as np
import scipy as sp
import test_module
import ciamod
import TPmod
import settings
import os
import gc
import sys
import pickle
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from mpi4py import MPI
import utils
from collections import namedtuple
from specops import proc_spec

  
def teffRM(theta,re_params,sigDist,sigPhot):


    args_instance=settings.runargs
    all_params,all_params_values =utils.get_all_parametres(re_params.dictionary) 
    params_master = namedtuple('params',all_params)
    params_instance = params_master(*theta)



    dist = args_instance.dist
    gnostics = 0
    trimspec, photspec,tauspec,cfunc = test_module.modelspec(theta,re_params,args_instance,gnostics) 
    # now calculate Fbol by summing the spectrum across its wave bins
    fbol = 0.0

    if re_params.samplemode=='multinest':
        M= params_instance.M
        R= params_instance.R* 69911e3
        GM = (6.67E-11 * M*1.898e27)
        logg = np.log10(100.* GM / R**2.)
        D = (args_instance.dist + (np.random.randn()*args_instance.dist_err)) * 3.086e16
        # D = dist * 3.086e16
        r2d2 = R**2. / D**2.

    if re_params.samplemode=='mcmc':
        r2d2= params_instance.r2d2

    wave,flux=proc_spec(inputspec=trimspec, theta=params_instance, re_params=re_params, args_instance=args_instance, do_scales=args_instance.do_scales,do_conv=False)
    flux[np.where(np.isnan(flux))[0]] = 0.0

    for j in range(1, (wave.size - 1)):
        sbin = ((wave[j] - wave[j-1]) + (wave[j+1] - wave[j])) / 2. 
        fbol = (sbin * flux[j]) + fbol

    # Get Lbol in log(L / Lsol)
    lsun = 3.828e26
    l_bol = np.log10((fbol * 4.*np.pi*(dist * 3.086e16)**2) / lsun)

    # now get T_eff
    t_ff = ((fbol/(r2d2 * 5.670367e-8))**(1./4.))

    # and Radius
    sigR2D2 = sigPhot * r2d2 * (-1./2.5)* np.log(10.)

    sigD = sigDist * 3.086e16
    D = dist * 3.086e16

    R = np.sqrt(((np.random.randn() * sigR2D2)+ r2d2)) \
        * ((np.random.randn()* sigD) + D)

    g = (10.**params_instance.logg)/100.

    # and mass

    M = (R**2 * g/(6.67E-11))/1.898E27
    R = R / 71492e3

    result = np.concatenate((theta,np.array([l_bol, t_ff,R,M])),axis=0)
    
    return result



    # # and Radius
    # sigR2D2 = sigPhot * r2d2 * (-1./2.5)* np.log(10.)

    # sigD = sigDist * 3.086e16
    # D = dist * 3.086e16

    # R = np.sqrt(((np.random.randn() * sigR2D2)+ r2d2)) \
    #     * ((np.random.randn()* sigD) + D)

    # g = (10.**logg)/100.

    # # and mass

    # M = (R**2 * g/(6.67E-11))/1.898E27
    # R = R / 71492e3

    # # Now lets get the C/O and M/H ratios...
    # # read molecules (elements!) to be used for M/H and C/O from theta
    # # Think about these choices, and maybe experiment
    # # Are the elements depleted by missed gases, or condensation
    # # e.g. Fe/H, from retrieved FeH abundance will look subsolar
    # # due to condensation of Fe.
    # # Similarly, N2 is not observable, so N/H from NH3 may look subsolar...

    # # Example here is for T dwarf gaslist:['h2o','ch4','co','co2','nh3','h2s','k','na']

    # # We will base e
    # h2o = params_instance.h2o
    # ch4 = params_instance.ch4
    # co = params_instance.co 
    # co2 = params_instance.co2
    # kna = params_instance.K_Na

    # # first get the C/O

    # O = 10**h2o + 10**co + 2.*10**(co2) 
    # C = 10**(co) + 10**(co2) + 10**(ch4)

    # CO_ratio = C/O

    # # rest of the elements
    # NaK = 10**kna

    # # Determine "fraction" of H2 in the L dwarf
    # gas_sum = 10**h2o + 10**co +10**co2 + 10**ch4 + 10**kna
    # fH2 = (1-gas_sum)* 0.84  # fH2/(fH2+FHe) = 0.84
    # fH = 2.*fH2

    
    # # Determine linear solar abundance sum of elements in our L dwarf
    # # abundances taken from Asplund+ 2009
    # solar_H = 12.00
    # solar_O = 10**(8.69-solar_H)
    # solar_C = 10**(8.43-solar_H) 
    # #solar_Ti = 10**(4.95-solar_H)
    # #solar_V = 10**(3.93-solar_H)
    # #solar_Cr = 10**(5.64-solar_H)
    # #solar_Fe = 10**(7.50-solar_H)
    # solar_NaK = 10**(6.24-solar_H) + 10**(5.03-solar_H)
    
    # # Calculate the metallicity fraction in the star and the same for the sun and then make the ratio
    # metallicity_target = (O/fH) + (C/fH) + (NaK/fH)
    # metallicity_sun = solar_O + solar_C + solar_NaK

    # MH = np.log10(metallicity_target / metallicity_sun)

    # result = np.concatenate((theta,np.array([l_bol, t_ff, R, M, MH, CO_ratio])),axis=0)
    
    # return result

def get_endchain(runname,fin):
    if (fin == 1):
        pic = runname+".pk1"
        with open(pic, 'rb') as input:
            sampler = pickle.load(input) 
        nwalkers = sampler.chain.shape[0]
        niter = sampler.chain.shape[1]
        ndim = sampler.chain.shape[2]
        flatprobs = sampler.lnprobability[:,:].reshape((-1))
        flatendchain = sampler.chain[:,niter-2000:,:].reshape((-1,ndim))
        flatendprobs = sampler.lnprobability[niter-2000:,:].reshape((-1))
 

    elif(fin ==0):
        pic = runname+"_snapshot.pic"
        with open(pic, 'rb') as input:
            chain,probs = pickle.load(input) 
        nwalkers = chain.shape[0]
        ntot = chain.shape[1]
        ndim = chain.shape[2]
        niter = int(np.count_nonzero(chain) / (nwalkers*ndim))
        flatprobs = probs[:,:].reshape((-1))
        flatendchain = chain[:,(niter-2000):niter,:].reshape((-1,ndim))
        flatendprobs = probs[(niter-2000):niter,:].reshape((-1))
    else:
        print("File extension not recognised")

        
    return flatendchain, flatendprobs,ndim

def getargs(runname):
    
    pic = runname+"_runargs.pic"
    with open(pic, 'rb') as input:
        gases_myP,chemeq,dist, cloudtype,do_clouds,gasnum,cloudnum,inlinetemps,coarsePress,press,inwavenum,linelist,cia,ciatemps,use_disort,fwhm,obspec,proftype,do_fudge, prof,do_bff,bff_raw,ceTgrid,metscale,coscale = pickle.load(input) 

    args =  gases_myP,chemeq,dist, cloudtype,do_clouds,gasnum,cloudnum,inlinetemps,coarsePress,press,cia,ciatemps,use_disort,fwhm,obspec,proftype,do_fudge, prof,do_bff,bff_raw,ceTgrid,metscale,coscale
    return args
