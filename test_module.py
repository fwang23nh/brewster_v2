#!/usr/bin/env python

""" Module of bits to define prior,lnlike,and lnprob for MCMC/Multinest """
from __future__ import print_function
import time
import math
import numpy as np
import scipy as sp
import gc
import ciamod
import TPmod
import os
import sys
import pickle
import forwardmodel
import cloud_dic_new
from builtins import str
from builtins import range
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian1DKernel
from bensconv import prism_non_uniform
from bensconv import conv_uniform_R
from bensconv import conv_uniform_FWHM
from bensconv import conv_non_uniform_R
from collections import namedtuple
import utils
import settings
import gas_nonuniform
import Priors
from rotBroadInt import rot_int_cmj as rotBroad


__author__ = "Fei Wang"
__copyright__ = "Copyright 2024 - Fei Wang"
__credits__ = ["Fei Wang", "Ben Burningham"]
__license__ = "GPL"
__version__ = "0.2"  
__maintainer__ = ""
__email__ = ""
__status__ = "Development"




def lnprob(theta,re_params):

    args_instance=settings.runargs
    Priors_instance =Priors.Priors(theta,re_params,args_instance.instrument)
    # now check against the priors, if not beyond them, run the likelihood
    lp = Priors_instance.priors
    if not np.isfinite(lp):
        return -np.inf
    # run the likelihood
    lnlike_value = lnlike(theta,re_params)

    lnprb = lp+lnlike_value
    if np.isnan(lnprb):
        lnprb = -np.inf
    return lnprb
                                          

def lnlike(theta,re_params):


    all_params,all_params_values =utils.get_all_parametres(re_params.dictionary) 
    #make instrument instance as input parameter
    #arg_instance, save R-file into arg_instance
    params_master = namedtuple('params',all_params)
    params_instance = params_master(*theta)
    args_instance=settings.runargs

    press=args_instance.press
    fwhm=args_instance.fwhm
    obspec=args_instance.obspec
    proftype=args_instance.proftype
    do_fudge=args_instance.do_fudge

    # get the spectrum
    # for MCMC runs we don't want diagnostics
    gnostics = 0
    shiftspec, photspec,tauspec,cfunc = modelspec(theta,re_params,args_instance,gnostics)


    # Get the scaling factors for the spectra. What is the FWHM? Negative number: preset combination of instruments
    if (fwhm < 0.0):
        if (fwhm == -1 or fwhm == -3 or fwhm == -4):
            scale1 =  params_instance.scale1 
            scale2 =  params_instance.scale2
            if (do_fudge == 1):
                logf = [params_instance.tolerance_parameter_1,params_instance.tolerance_parameter_2,params_instance.tolerance_parameter_3] #theta[ng+5:ng+8]
            else:
                # This is a place holder value so the code doesn't break
                logf = np.log10(0.1*(max(obspec[2,10::3]))**2)
        elif (fwhm == -2):
            scale1 = params_instance.r2d2
            if (do_fudge == 1):
                logf = [params_instance.tolerance_parameter_1,params_instance.tolerance_parameter_2]
            else:
                # This is a place holder value so the code doesn't break
                logf = np.log10(0.1*(max(obspec[2,10::3]))**2)
        elif (fwhm == -5):
            if (do_fudge == 1):
                logf = params_instance.tolerance_parameter_1
            else:
                # This is a place holder value so the code doesn't break
                logf = np.log10(0.1*(max(obspec[2,10::3]))**2)
        elif (fwhm == -6):
            if (do_fudge == 1):
                logf = params_instance.tolerance_parameter_1
            else:
                # This is a place holder value so the code doesn't break
                logf = np.log10(0.1*(max(obspec[2,10::3]))**2)
    elif (fwhm == 888):
        if (do_fudge ==1):
            logf = [params_instance.tolerance_parameter_1,params_instance.tolerance_parameter_2]
        else:
            # This is a place holder value so the code doesn't break
            logf = np.log10(0.1*(max(obspec[2,10::3]))**2)
    elif (fwhm == 777):
        if (do_fudge ==1):
            logf = params_instance.frac_param
        else:
            # This is a place holder value so the code doesn't break
            logf = np.log10(0.1*(max(obspec[2,10::3]))**2)
    elif (fwhm == 555):
        scales_param = args_instance.scales
        nonzero_scales = sorted(set(scales_param) - {0}) 

        scales = {}
        for i in nonzero_scales:
            pname = f"scale{i}"
            if hasattr(params_instance, pname): #it just checks if it exists, return True boolean
                scales[pname] = getattr(params_instance, pname) #returns actual value
        if (do_fudge == 1):
            if np.max(args_instance.logf_flag) == 1.0:
                logf = [params_instance.tolerance_parameter_1]
            elif np.max(args_instance.logf_flag) == 2.0:
                logf = [params_instance.tolerance_parameter_1,params_instance.tolerance_parameter_2]
            elif np.max(args_instance.logf_flag) == 3.0:
                logf = [params_instance.tolerance_parameter_1, params_instance.tolerance_parameter_2, params_instance.tolerance_parameter_3]
            elif np.max(args_instance.logf_flag) == 4.0:
                logf = [params_instance.tolerance_parameter_1,params_instance.tolerance_parameter_2,params_instance.tolerance_parameter_3,params_instance.tolerance_parameter_4]
        else:
            # This is a place holder value so the code doesn't break
            logf = np.log10(0.1*(max(obspec[2,10::3]))**2)
                
    else:
        if (do_fudge == 1):
            logf = params_instance.tolerance_parameter_1
        else:
            # This is a place holder value so the code doesn't break
            logf = np.log10(0.1*(max(obspec[2,10::3]))**2)

    modspec = np.array([shiftspec[0,::-1],shiftspec[1,::-1]])

    if hasattr(params_instance, "vrad"):
        rotspec = rotBroad(modspec[0],modspec[1],params_instance.vsini)
        modspec[1,:] = rotspec


    # If we've set a value for FWHM that we're using...
    if (fwhm > 0.00 and fwhm < 1.00):
        # this is a uniform FWHM in microns
        spec = conv_uniform_FWHM(obspec,modspec,fwhm)
        if (do_fudge == 1):
            s2=obspec[2,:]**2 + 10.**logf
        else:
            s2 = obspec[2,:]**2

        lnLik=-0.5*np.sum((((obspec[1,:] - spec[:])**2) / s2) + np.log(2.*np.pi*s2))       
    elif (fwhm > 10.00 and fwhm < 500):
        # this is a uniform resolving power R.
        Res = fwhm
        spec = conv_uniform_R(obspec,modspec,Res)
        if (do_fudge == 1):
            s2=obspec[2,:]**2 + 10.**logf
        else:
            s2 = obspec[2,:]**2

        lnLik=-0.5*np.sum((((obspec[1,:] - spec[:])**2) / s2) + np.log(2.*np.pi*s2))
    elif (fwhm == 0.0):
        # Use convolution for Spex
        spec = prism_non_uniform(obspec,modspec,3.3)
        if (do_fudge == 1):
            s2=obspec[2,::3]**2 + 10.**logf
        else:
            s2 = obspec[2,::3]**2

        lnLik=-0.5*np.sum((((obspec[1,::3] - spec[::3])**2) / s2) + np.log(2.*np.pi*s2))
    elif (fwhm == 1.0):
        # Use convolution for JWST-NIRSpec PRISM
        spec = prism_non_uniform(obspec,modspec,2.2)
        if (do_fudge == 1):
            s2=obspec[2,::2]**2 + 10.**logf
        else:
            s2 = obspec[2,::2]**2

        lnLik=-0.5*np.sum((((obspec[1,::2] - spec[::2])**2) / s2) + np.log(2.*np.pi*s2))
    elif (fwhm == 2.0):
        # combo of JWST-NIRSpec PRISM + G395H grism
        # single scaling & single fudge factor
        spec = np.zeros_like(obspec[0,:])
        # first convolution for JWST-NIRSpec PRISM
        or1  = np.where(obspec[0,:] < 2.9)
        spec[or1] = prism_non_uniform(obspec[:,or1],modspec,2.2)
        # now 1st grism bit
        dL = 0.0015
        or2  = np.where(np.logical_and(obspec[0,:] > 2.9,obspec[0,:] < 3.69))
        spec[or2] =  conv_uniform_FWHM(obspec[:,or2],modspec,dL)
        # a bit more prism
        or3 = np.where(np.logical_and(obspec[0,:] > 3.69,obspec[0,:] < 3.785))
        spec[or3] = prism_non_uniform(obspec[:,or3],modspec,2.2)
        # 2nd bit of grism
        or4 = np.where(np.logical_and(obspec[0,:] > 3.785,obspec[0,:] < 5.14))
        spec[or4] =  conv_uniform_FWHM(obspec[:,or4],modspec,dL)
        # the rest of prism
        or5 = np.where(obspec[0,:] > 5.14)
        spec[or5] = prism_non_uniform(obspec[:,or5],modspec,2.2)
        if (do_fudge == 1):
            s2=obspec[2,:]**2 + 10.**logf
        else:
            s2 = obspec[2,:]**2

        lnLik=-0.5*np.sum((((obspec[1,:] - spec[:])**2) / s2) + np.log(2.*np.pi*s2))
        
    elif (fwhm == 888):
        #Non-uniform R, NIRSpec + MIRI, two tolerance parameters
        print(f"obspec shape: {obspec.shape}")
        print(f"modspec shape: {modspec.shape}")
        print(f"modspec[1] shape: {modspec[1].shape}") 
        print(f"modspec[0] shape: {modspec[0].shape}")


        or1 = np.where(obspec[0,:]<4.634)
        #print('or1.shape', or1.shape)
        or2 = np.where(obspec[0,:]>4.634)
       # print('modspec[1, or2].shape: ',modspec[1, or2].shape
        #print('or2.shape', or2.shape)
        print('modspec[1,or2] shape: ',modspec[1, or2].shape)
        print('obspec [0,or2] shape: ',obspec[0, or2].shape)
        
        spec1 = conv_non_uniform_R(modspec[1,or1], modspec[0,or1], args_instance.R, obspec[0,or1])
        
        spec2 = conv_non_uniform_R(modspec[1,or2], modspec[0,or2], args_instance.R, obspec[0,or2])


        if (do_fudge == 1):
            print(f"logf type: {type(logf)}, shape: {np.shape(logf)}")

            s2 = obspec[2,or1]**2 + 10.**logf[0]
            s3 = obspec[2,or2]**2 + 10.**logf[1]
        else:
            s2 = obspec[2,or1]**2
            s3 = obspec[2,or2]**2 
        
        lnLik1=-0.5*np.sum((((obspec[1,or1] - spec1[:])**2) / s2) + np.log(2.*np.pi*s2)) 
        lnLik2=-0.5*np.sum((((obspec[1,or2] - spec2[:])**2) / s3) + np.log(2.*np.pi*s3))
        lnLik = lnLik1 + lnLik2
        
        
        
    elif(fwhm == 777): #STILL NEEDS TO BE TESTED
        #Non_uniform R, tolerance parameter as a fraction of error,  so we are allowing the tolerance parameter to be different at each datapoint
        
        spec1 = conv_non_uniform_R(modspec[1,:], modspec[0,:], args_instance.R, obspec[0,:])
        
        if (do_fudge == 1):
            s2 = obspec[2,:]**2 + (params_instance.frac_param*obspec[2,:])**2
        else:
            s2 = obspec[2,:]**2
            
        lnLik1=-0.5*np.sum((((obspec[1,:] - spec1[:])**2) / s2) + np.log(2.*np.pi*s2)) 
        
        
    elif(fwhm == 555):
        #Non-uniform R, the user provides the R file with flags for the number and location of the tolerance and scale parameters
        #the columns of the R file as follows: R, wl, tol_flag,scale_flag

        log_f_param = args_instance.logf_flag
        log_f_param_max = int(np.max(log_f_param))
       
        scales_param = args_instance.scales
        scales_param_max = int(np.max(scales_param))
       
        lnLik = 0.0
       
        region_flags = np.unique(np.vstack((log_f_param, scales_param)).T, axis=0)#get unique values as a 2 column array [logf,scales]
        #for i,j in region_flags:
        for logf_flag_val, scale_flag_val in region_flags: #loop thru them, so we get each flags
            or_indices = np.where( (log_f_param == logf_flag_val) & (scales_param == scale_flag_val) ) #getting wl regions where both conditions are met

            obs_wl_i = obspec[0, :]
            spec_i = conv_non_uniform_R(modspec[1, :], modspec[0, :], args_instance.R[or_indices], obs_wl_i[or_indices])

        # IF THERE ARE SCALE PARAMETERS
            if scale_flag_val > 0:
                scale_name = f"scale{int(scale_flag_val)}"
                if scale_name in params_instance._fields:
                    scale_value = getattr(params_instance, scale_name)
                    spec_i = scale_value * spec_i  

            if (do_fudge == 1) and (logf_flag_val > 0):
                s_i = obspec[2, or_indices]**2 + 10.**logf[int(logf_flag_val)-1]
            else:
                s_i = obspec[2, or_indices]**2

            lnLik_i = -0.5 * np.sum(((obspec[1, or_indices] - spec_i[:])**2) / s_i + np.log(2.*np.pi*s_i))
            lnLik += lnLik_i
       
       
            
    elif (fwhm < 0.0):
        lnLik = 0.0
        # This is for multi-instrument cases
        # -1: spex + akari + IRS
        # -2: spex + IRS
        # -3: spex + Lband + IRS
        if (fwhm == -1):

            # Spex
            or1  = np.where(obspec[0,:] < 2.5)
            spec1 = prism_non_uniform(obspec[:,or1],modspec,3.3)

            # AKARI IRC
            # dispersion constant across order 0.0097um
            # R = 100 at 3.6um for emission lines
            # dL ~constant at 3.6 / 120
            dL = 0.03
            or2 = np.where(np.logical_and(obspec[0,:] > 2.5,obspec[0,:] < 5.0))
            spec2 = scale1 * conv_uniform_FWHM(obspec[:,or2],modspec,dL)

            # Spitzer IRS
            # R roughly constant within orders, and orders both appear to
            # have R ~ 100
            R = 100.0
            or3 = np.where(obspec[0,:] > 5.0)
            spec3 = scale2 * conv_uniform_R(obspec[:,or3],modspec,R)

            if (do_fudge == 1):
                s1 = obspec[2,or1]**2 + 10.**logf[0]
                s2 = obspec[2,or2]**2 + 10.**logf[1]
                s3 = obspec[2,or3]**2 + 10.**logf[2]
            else:
                s1 = obspec[2,or1]**2
                s2 = obspec[2,or2]**2
                s3 = obspec[2,or3]**2


            lnLik1=-0.5*np.sum((((obspec[1,or1] - spec1)**2) / s1) + np.log(2.*np.pi*s1))
            lnLik2=-0.5*np.sum((((obspec[1,or2] - spec2)**2) / s2) + np.log(2.*np.pi*s2))
            lnLik3=-0.5*np.sum((((obspec[1,or3] - spec3)**2) / s3) + np.log(2.*np.pi*s3))
            lnLik = lnLik1 + lnLik2 + lnLik3

        elif (fwhm == -2):
            # This is just spex + IRS
            # Spex
            or1  = np.where(obspec[0,:] < 2.5)
            spec1 = prism_non_uniform(obspec[:,or1],modspec,3.3)

            # Spitzer IRS
            # R roughly constant within orders, and orders both appear to
            # have R ~ 100
            R = 100.0
            or3 = np.where(obspec[0,:] > 5.0)
            spec3 = scale1 * conv_uniform_R(obspec[:,or3],modspec,R)

            if (do_fudge == 1):
                s1 = obspec[2,or1]**2 + 10.**logf[0]
                s3 = obspec[2,or3]**2 + 10.**logf[1]
            else:
                s1 = obspec[2,or1]**2
                s3 = obspec[2,or3]**2


            lnLik1=-0.5*np.sum((((obspec[1,or1] - spec1)**2) / s1) + np.log(2.*np.pi*s1))
            lnLik3=-0.5*np.sum((((obspec[1,or3] - spec3)**2) / s3) + np.log(2.*np.pi*s3))
            lnLik = lnLik1 + lnLik3
            
        elif (fwhm == -3):
            # This is spex + Mike Cushing's L band R = 425 + IRS
            # Spex
            or1  = np.where(obspec[0,:] < 2.5)
            spec1 = prism_non_uniform(obspec[:,or1],modspec,3.3)

            # Mike Cushing supplied L band R = 425
            # dispersion constant across order 0.0097um
            # R = 425
            R = 425
            or2 = np.where(np.logical_and(obspec[0,:] > 2.5,obspec[0,:] < 5.0))
            spec2 = scale1 * conv_uniform_R(obspec[:,or2],modspec,R)

            # Spitzer IRS
            # R roughly constant within orders, and orders both appear to
            # have R ~ 100
            R = 100.0
            or3 = np.where(obspec[0,:] > 5.0)
            spec3 = scale2 * conv_uniform_R(obspec[:,or3],modspec,R)

            if (do_fudge == 1):
                s1 = obspec[2,or1]**2 + 10.**logf[0]
                s2 = obspec[2,or2]**2 + 10.**logf[1]
                s3 = obspec[2,or3]**2 + 10.**logf[2]
            else:
                s1 = obspec[2,or1]**2
                s2 = obspec[2,or2]**2
                s3 = obspec[2,or3]**2


            lnLik1=-0.5*np.sum((((obspec[1,or1] - spec1)**2) / s1) + np.log(2.*np.pi*s1))
            lnLik2=-0.5*np.sum((((obspec[1,or2] - spec2)**2) / s2) + np.log(2.*np.pi*s2))
            lnLik3=-0.5*np.sum((((obspec[1,or3] - spec3)**2) / s3) + np.log(2.*np.pi*s3))
            lnLik = lnLik1 + lnLik2 + lnLik3
            
        elif (fwhm == -4):
            # This is spex + GNIRS L band R = 600 + IRS
            # Spex
            or1  = np.where(obspec[0,:] < 2.5)
            spec1 = prism_non_uniform(obspec[:,or1],modspec,3.3)

            # Katelyn Allers spectrum of GNIRS R = 600
            # R = 600 @ 3.5um linearly increading across order
            # i.e. FWHM - 0.005833
            dL = 0.005833
            #dL = 0.0097

            or2 = np.where(np.logical_and(obspec[0,:] > 2.5,obspec[0,:] < 5.0))
            spec2 = scale1 * conv_uniform_FWHM(obspec[:,or2],modspec,dL)

            # Spitzer IRS
            # R roughly constant within orders, and orders both appear to
            # have R ~ 100
            R = 100.0
            #mr3 = np.where(modspec[0,:] > 5.0)
            or3 = np.where(obspec[0,:] > 5.0)
            spec3 = scale2 * conv_uniform_R(obspec[:,or3],modspec,R)

            if (do_fudge == 1):
                s1 = obspec[2,or1]**2 + 10.**logf[0]
                s2 = obspec[2,or2]**2 + 10.**logf[1]
                s3 = obspec[2,or3]**2 + 10.**logf[2]
            else:
                s1 = obspec[2,or1]**2
                s2 = obspec[2,or2]**2
                s3 = obspec[2,or3]**2


            lnLik1=-0.5*np.sum((((obspec[1,or1] - spec1)**2) / s1) + np.log(2.*np.pi*s1))
            lnLik2=-0.5*np.sum((((obspec[1,or2] - spec2)**2) / s2) + np.log(2.*np.pi*s2))
            lnLik3=-0.5*np.sum((((obspec[1,or3] - spec3)**2) / s3) + np.log(2.*np.pi*s3))
            lnLik = lnLik1 + lnLik2 + lnLik3

        elif (fwhm == -5):
            # This is JWST NIRSpec + MIRI MRS no scaling + 1 fudge
            join = np.array([0.,5.1,5.7,7.59,11.6,13.4,15.49,18.01,20.0])
            pix = np.array([2.2,1.9,2.0,2.2,2.4,3.1,3.0,3.3])

            # Now we just work through the Prism +MRS orders,
            # using mid point in overlap regions
            # divided into chunk based on fwhm of res element in pixels
            spec = np.zeros_like(obspec[0,:])
                                 
            for i in range(0,pix.size):
                bit = np.where(np.logical_and(obspec[0,:] > join[i],obspec[0,:] < join[i+1]))
                spec[bit] = prism_non_uniform(obspec[:,bit],modspec,pix[i])

         
            if (do_fudge == 1):
                s2 = obspec[2,:]**2 + 10.**logf
            else:
                s2 = obspec[2,:]**2

            lnLik=-0.5*np.sum((((obspec[1,:] - spec)**2) / s2) + np.log(2.*np.pi*s2))

        elif (fwhm == -6):
            # This is UKIRT orders 1 and 2 based on Geballe 1996 cuts 
            # Second Order                           
            # R ~ 780 x Lambda (linear increase across order)
            # Order 2 (0.95 - 1.40 um)
            # FWHM ~ 1.175/780 = 0.001506    
            dL1 = 0.001506
            or1  = np.where(obspec[0,:] < 1.585)

            spec1 = conv_uniform_FWHM(obspec[:,or1],modspec,dL1)

            # First Order                            
            # R ~ 390 x Lambda (linear increase across order)
            # Order 1 (1.30 - 5.50 um) 
            # FWHM ~ 3.4/390 = 0.008717
            dL2 = 0.008717
            or2 = np.where(obspec[0,:] > 1.585)

            spec2 = conv_uniform_FWHM(obspec[:,or2],modspec,dL2)

            if (do_fudge == 1):
                s1 = obspec[2,or1]**2 + 10.**logf
                s3 = obspec[2,or2]**2 + 10.**logf
            else:
                s1 = obspec[2,or1]**2
                s3 = obspec[2,or2]**2


            lnLik1=-0.5*np.sum((((obspec[1,or1[0][::7]] - spec1[::7])**2) / s1[0][::7]) + np.log(2.*np.pi*s1[0][::7]))
            lnLik3=-0.5*np.sum((((obspec[1,or2[0][::3]] - spec2[::3])**2) / s3[0][::3]) + np.log(2.*np.pi*s3[0][::3]))
            lnLik = lnLik1 + lnLik3

        elif (fwhm == -7):
            #This is CGS4 NIR + NIRC Lband * CGS4 Mband
            # CGS4 Second order R = 780xLambda
            dL1 = 0.001506
            or1 = np.where(obspec[0, :] < 1.585)
            spec1 = conv_uniform_FWHM(obspec[:, or1], modspec, dL1)

            # CGS4 First order R = 390xLambda
            dL2 = 0.008717
            or2 = np.where(np.logical_and(obspec[0, :] > 1.585, obspec[0, :] < 2.52))
            spec2 = conv_uniform_FWHM(obspec[:, or2], modspec, dL2)

            # Oppenheimer 1998 NIRC L band spectrum
            ###EDIT### Central wavelength @ 3.492 with FWHM=1.490 for lw band
            # Using R=164
            #dL3 = 0.0213
            R=164.0
            or3 = np.where(np.logical_and(obspec[0, :] > 2.52, obspec[0, :] < 4.15))
            spec3 = scale1 * conv_uniform_R(obspec[:, or3], modspec, R)

            # CGS4 M band
            # Order 1 using 1".2 slit, 75 line/mm grating, 150 mm focal length camera
            ###EDIT### R=400xLambda
            dL4 = 0.0085
            or4 = np.where(obspec[0, :] > 4.15)
            spec4 = scale2 * conv_uniform_FWHM(obspec[:, or4], modspec, dL4)

            if (do_fudge == 1):
                s1 = obspec[2, or1] ** 2 + 10. ** logf[0]
                s2 = obspec[2, or2] ** 2 + 10. ** logf[0]
                s3 = obspec[2, or3] ** 2 + 10. ** logf[1]
                s4 = obspec[2, or4] ** 2 + 10. ** logf[2]
            else:
                s1 = obspec[2, or1] ** 2
                s2 = obspec[2, or2] ** 2
                s3 = obspec[2, or3] ** 2
                s4 = obspec[2, or4] ** 2  
  
            lnLik1 = -0.5 * np.sum((((obspec[1, or1[0][::7]] - spec1[::7]) ** 2) / s1[0][::7]) + np.log(2. * np.pi * s1[0][::7]))
            lnLik2 = -0.5 * np.sum((((obspec[1, or2[0][::3]] - spec2[::3]) ** 2) / s2[0][::3]) + np.log(2. * np.pi * s2[0][::3]))
            lnLik3 = -0.5 * np.sum((((obspec[1, or3] - spec3) ** 2) / s3) + np.log(2. * np.pi * s3))
            lnLik4 = -0.5 * np.sum((((obspec[1, or4] - spec4) ** 2) / s4) + np.log(2. * np.pi * s4))

            lnLik = lnLik1 + lnLik2 + lnLik3 + lnLik4

    if np.isnan(lnLik):
        lnLik = -np.inf

    samplemode=re_params.samplemode.lower()
    if samplemode=="multinest":
        if proftype==1 or proftype==77:
            intemp_keys = list(re_params.dictionary['pt']['params'].keys())
            intemp = np.array([getattr(params_instance, key) for key in intemp_keys])
            T=intemp[1:]

            if proftype==77:
                T = np.empty([press.size])
                T[:] = -100.
                delta= np.exp(params_instance.lndelta)
                alpha=params_instance.alpha
                P1 = ((1/delta)**(1/alpha))
                # put prior on P1 to put it shallower than 100 bar   
                if  (1 < alpha  < 2. and P1 < 100. and P1 > press[0]
                    and params_instance.T1 > 0.0 and params_instance.T2 > 0.0 and params_instance.T3 > 0.0 and params_instance.Tint >0.0):
                    T = TPmod.set_prof(proftype,junkP,press,intemp) # allow inversion 

            # bits for smoothing in prior
            gam = params_instance.gamma

            if (gam>0 and  (min(T) > 1.0) and (max(T) < 6000.)):
                diff=np.roll(T,-1)-2.*T+np.roll(T,1)
                pp=len(T)
                logbeta = -5.0
                beta=10.**logbeta
                alpha=1.0
                x=gam
                invgamma=((beta**alpha)/math.gamma(alpha)) * (x**(-alpha-1)) * np.exp(-beta/x)
                prprob = (-0.5/gam)*np.sum(diff[1:-1]**2) - 0.5*pp*np.log(gam) + np.log(invgamma)

                lnLik+=prprob
        
    return lnLik


def modelspec(theta,re_params,args_instance,gnostics):


    all_params,all_params_values =utils.get_all_parametres(re_params.dictionary) 
    params_master = namedtuple('params',all_params)
    params_instance = params_master(*theta)


    # Unpack all necessary parameters into local variables
    press=args_instance.press
    fwhm=args_instance.fwhm
    obspec=args_instance.obspec
    proftype=args_instance.proftype
    do_fudge=args_instance.do_fudge
    #gpoints=args_instance.gpoints
    #weights=rgs_instance.weights
 
        
    nlayers = press.size
    if args_instance.chemeq == 0:
        gas_keys = re_params.dictionary['gas'].keys()
        gas_keys=list(gas_keys)
        invmr=np.array([getattr(params_instance, key) for key in gas_keys])


    else:
        mh  = params_instance.mh
        co =  params_instance.co

        mfit = interp1d(args_instance.metscale,args_instance.gases_myP,axis=0)
        gases_myM = mfit(mh)
        cfit = interp1d(args_instance.coscale,gases_myM,axis=0)
        invmr = cfit(co)


    if re_params.samplemode=='multinest':
        M= params_instance.M
        R= params_instance.R* 69911e3
        GM = (6.67E-11 * M*1.898e27)
        logg = np.log10(100.* GM / R**2.)
        D = (args_instance.dist + (np.random.randn()*args_instance.dist_err)) * 3.086e16
        # D = dist * 3.086e16
        R2D2 = R**2. / D**2.

    
    if (fwhm < 0.0):
        if (fwhm == -1 or fwhm == -3 or fwhm == -4):
            if re_params.samplemode=='mcmc':
                r2d2 = [params_instance.r2d2,params_instance.scale1,params_instance.scale2]  #theta[ng+1:ng+4]
            dlam = params_instance.dlambda
            # if (do_fudge == 1):
            #     logf =[params_instance.tolerance_parameter_1,params_instance.tolerance_parameter_2,params_instance.tolerance_parameter_3] #theta[ng+5:ng+8]
            # else:
            #     # This is a place holder value so the code doesn't break
            #     logf = np.log10(0.1*(max(obspec[2,10::3]))**2)

        elif (fwhm == -2):
            if re_params.samplemode=='mcmc':
                r2d2 = [params_instance.r2d2,params_instance.scale1]  #theta[ng+1:ng+3]
            dlam = params_instance.dlambda
            # if (do_fudge == 1):
            #     logf =[params_instance.tolerance_parameter_1,params_instance.tolerance_parameter_2] # theta[ng+4:ng+6]
            # else:
            #     # This is a place holder value so the code doesn't break
            #     logf = np.log10(0.1*(max(obspec[2,10::3]))**2)

        elif (fwhm == -5):
            if re_params.samplemode=='mcmc':
                r2d2 = params_instance.r2d2
            dlam = params_instance.dlambda
            # if (do_fudge == 1):
            #     logf = params_instance.tolerance_parameter_1
            # else:
            #     # This is a place holder value so the code doesn't break
            #     logf = np.log10(0.1*(max(obspec[2,10::3]))**2)

        elif (fwhm == -6):
            if re_params.samplemode=='mcmc':
                r2d2 = params_instance.r2d2
            dlam = params_instance.dlambda
            # if (do_fudge == 1):
            #     logf = params_instance.tolerance_parameter_1
            # else:
            #     # This is a place holder value so the code doesn't break                                                                      
            #     logf = np.log10(0.1*(max(obspec[2,10::3]))**2)

    else:
        if re_params.samplemode=='mcmc':
            r2d2 = params_instance.r2d2
        dlam = params_instance.dlambda
        # if (do_fudge == 1):
        #     logf = params_instance.tolerance_parameter_1
        # else:
        #     # This is a place holder value so the code doesn't break
        #     logf = np.log10(0.1*(max(obspec[2,10::3]))**2)


        
    npatches = args_instance.cloudmap.shape[0]
    if (npatches > 1):
        prat =  params_instance.fcld
        pcover = np.array([prat,(1.-prat)])

    else:
        pcover = 1.0
        
    # use correct unpack method depending on situation
    # if ((npatches > 1) and np.all(do_clouds != 0)):
    #     cloudparams = cloud_dic.unpack_patchy(re_params,params_instance,cloudtype,cloudflag,do_clouds)
    # else:
    #     cloudparams = cloud_dic.unpack_default(re_params,params_instance,cloudtype,cloudflag,do_clouds)


    cloudparams=cloud_dic_new.cloud_unpack(re_params,params_instance)
    ndim = len(theta)

    intemp_keys = list(re_params.dictionary['pt']['params'].keys())
    intemp = np.array([getattr(params_instance, key) for key in intemp_keys])

    if (proftype == 1):
        gam = params_instance.gamma
        intemp=intemp[1:]

    elif (proftype == 77):
        gam = params_instance.gamma
        intemp=intemp[1:]
    elif (proftype == 2 or proftype ==3 or proftype ==7):
        intemp=intemp
    elif (proftype == 9):
        intemp = args_instance.prof

    else:
        raise ValueError("not valid profile type %s" %proftype)

    # set the profile
    temp = TPmod.set_prof(proftype,args_instance.coarsePress,press,intemp)

    ngas = len(args_instance.gaslist)
    bff = np.zeros([3,nlayers],dtype="float64")


    # check if its a fixed VMR or a profile from chem equilibrium
    # VMR is log10(VMR) !!!
    if args_instance.chemeq == 1:
        # this case is a profile
        ng = invmr.shape[2]
        ngas = ng - 3
        logVMR = np.zeros([ngas,nlayers],dtype='d')
        for p in range(0,nlayers):
            for g in range(0,ng):
                tfit = InterpolatedUnivariateSpline(args_instance.ceTgrid,invmr[:,p,g])
                if (g < 3):
                    bff[g,p] = tfit(temp[p])
                else:
                    logVMR[g-3,p]= tfit(temp[p])
        

    else:
        # This case is fixed VMR
        # chemeq = 0
        logVMR = np.empty((ngas,nlayers),dtype='d')
        alkratio = 16.2 #  from Asplund et al (2009)

        tmpvmr = np.empty(ngas,dtype='d')
        if (args_instance.gaslist[len(args_instance.gaslist)-1] == 'na' and args_instance.gaslist[len(args_instance.gaslist)-2] == 'k'):
            tmpvmr[0:(ngas-2)] = invmr[0:-1]
            tmpvmr[ngas-2] = np.log10(10.**invmr[-1] / (alkratio+1.)) # K
            tmpvmr[ngas-1] = np.log10(10.**invmr[-1] * (alkratio / (alkratio+1.))) # Na
        elif (args_instance.gaslist[len(args_instance.gaslist)-1] == 'cs'):
            #f values are ratios between Na and (K+Cs) and K and Cs respectively
            f1 = 1.348
            f2 = 8912.5
            tmpvmr[0:(ngas-3)] = invmr[0:-1]
            tmpvmr[ngas-1] = np.log10(10.**invmr[-1] / ((f1+1)*(f2+1))) # Cs
            tmpvmr[ngas-2] = np.log10(10.**invmr[-1] * (f1 /(f1+1)) ) # Na
            tmpvmr[ngas-3] = np.log10(10.**invmr[-1] - 10.**tmpvmr[ngas-2] - 10.**tmpvmr[ngas-1]) #K
        else:
            tmpvmr[0:ngas] = invmr[0:ngas]
            
        for i in range(0,ngas):
            logVMR[i,:] = tmpvmr[i]

        # # set H- in the high atmosphere    
        # if (gaslist[gasnum.size-1] == 'hmins'):
        #     logVMR[ngas-1,0:P_hmins_index]=tmpvmr[ngas-1]
        #     logVMR[ngas-1,P_hmins_index:]=-100

        # # set non-uniform gas profile 
        
        gastype_values = [info['gastype'] for key, info in re_params.dictionary['gas'].items() if 'gastype' in info]
            
        for i in range(len(gastype_values)):
            if  gastype_values[i]=="N":
                P_gas= getattr(params_instance, "p_ref_%s"%gas_keys[i])
                gas_alpha= getattr(params_instance, "alpha_%s"%gas_keys[i])
                t_gas= getattr(params_instance, gas_keys[i])
                gas_profile=gas_nonuniform.non_uniform_gas(press,P_gas,t_gas,gas_alpha)
                logVMR[i]=gas_profile

            elif gastype_values[i]=="H":

                p_gas= getattr(params_instance, "p_ref_%s"%gas_keys[i])
                P_gas = 10**p_gas

                if np.size(np.where((press>=P_gas))[0])==0:   #may return null array
                    p_gas_index=np.size(press)-1
                else:
                    p_gas_index=np.where((press>=P_gas))[0][0] 

                logVMR[i,0:p_gas_index]=tmpvmr[i]
                logVMR[i,p_gas_index:]=-100


    # now need to translate cloudparams in to cloud profile even
    # if do_clouds is zero..

    # cloudprof,cloudrad,cloudsig = cloud_dic.atlas(do_clouds,cloudflag,cloudtype,cloudparams,press)

    cloudprof,cloudrad,cloudsig = cloud_dic_new.atlas(re_params,cloudparams,press)



    cloudprof = np.asfortranarray(cloudprof,dtype = 'float64')
    cloudrad = np.asfortranarray(cloudrad,dtype = 'float64')
    cloudsig = np.asfortranarray(cloudsig,dtype = 'float64')
    pcover = np.asfortranarray(pcover,dtype = 'float32')
    cloudsize = np.asfortranarray(args_instance.cloudsize,dtype = 'i')
    cloudmap = np.asfortranarray(args_instance.cloudmap,dtype = 'i')


    # do_clouds = np.asfortranarray(do_clouds,dtype = 'i')

    # get r2d2 sorted for multi-instruments
    if re_params.samplemode=='mcmc':
        logg = params_instance.logg
        if (fwhm < 0.0):
            if (fwhm == -1 or fwhm == -3 or fwhm == -4):
                R2D2 = r2d2[0]
                scale1 = r2d2[1]
                scale2 = r2d2[2]
            elif (fwhm == -2):
                R2D2 = r2d2[0]
                scale1 = r2d2[1]
            elif (fwhm == -5):
                R2D2 = r2d2
            elif (fwhm == -6):
                R2D2 = r2d2
        else:
            R2D2 = r2d2


    # Now get the BFF stuff sorted
    if (args_instance.chemeq == 0 and args_instance.do_bff == 1):
        for gas in range(0,3):
            for i in range(0,nlayers):
                tfit = InterpolatedUnivariateSpline(args_instance.ceTgrid,args_instance.bff_raw[:,i,gas],k=1)
                bff[gas,i] = tfit(temp[i])

        if (args_instance.gaslist[len(args_instance.gaslist)-1] == 'hmins'):
            bff[2,0:p_gas_index] = -50

    bff = np.asfortranarray(bff, dtype='float64')
    press = np.asfortranarray(press,dtype='float32')
    temp = np.asfortranarray(temp,dtype='float64')
    logVMR = np.asfortranarray(logVMR,dtype='float64')
    
    # Diagnostics below.
    # make_cf = get a contribution function
    # clphot = get pressure for cloud_tau = 1.0 as function of wavelength
    # ^^ i.e the cloud photosphere
    # ophot = get pressures for tau(not cloud) = 1.0 as function of wavelength]
    # ^^ i.e. the photosphere due to other (gas phase) opacities)

    # Set clphot,ophot and cfunc as we don't need these in the emcee run
    if (gnostics == 0):
        clphot = 0
        ophot = 0
        make_cf = 0
    else:
        clphot = 1
        ophot = 1
        make_cf = 1



    #now we can call the forward model
    outspec,tmpclphotspec,tmpophotspec,cf = forwardmodel.marv(temp,logg,R2D2,args_instance.gasnames,args_instance.gasmass,logVMR,pcover,cloudmap,args_instance.cloud_opaname,cloudsize,args_instance.cloudata,args_instance.miewave,args_instance.mierad,cloudrad,cloudsig,cloudprof,args_instance.inlinetemps,press,args_instance.inwavenum,settings.linelist,settings.cia,args_instance.ciatemps,args_instance.use_disort,clphot,ophot,make_cf,args_instance.do_bff,bff)

    # Trim to length where it is defined.
    nwave = args_instance.inwavenum.size
    trimspec = np.zeros([2,nwave],dtype='d')
    trimspec = outspec[:,:nwave]
    cloud_phot_press = tmpclphotspec[0:npatches,:nwave].reshape(npatches,nwave)
    other_phot_press = tmpophotspec[0:npatches,:nwave].reshape(npatches,nwave)
    cfunc = np.zeros([npatches,nwave,nlayers],dtype='d')
    cfunc = cf[:npatches,:nwave,:nlayers].reshape(npatches,nwave,nlayers)

    # now shift wavelen by delta_lambda
    shiftspec = np.empty_like(trimspec)

    if hasattr(params_instance, "vrad"):
        vrad = params_instance.vrad
        dlam = trimspec[0,:] * vrad/3e5

    shiftspec[0,:] =  trimspec[0,:] + dlam
    shiftspec[1,:] =  trimspec[1,:]

    # print("VMR")
    # print(logVMR)
    # print("----------------")
    # print("temp")
    # print(temp)
    # print("----------------")
    # print("cloudrad,cloudsig,cloudprof")
    # print(cloudrad,cloudsig,cloudprof)

    return shiftspec, cloud_phot_press,other_phot_press,cfunc



