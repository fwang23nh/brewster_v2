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
from specops import proc_spec


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
    Priors_instance =Priors.Priors(theta,re_params,args_instance)
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
    obspec=args_instance.obspec
    proftype=args_instance.proftype
    do_fudge=args_instance.do_fudge
    do_shift=args_instance.do_shift
    do_scales=args_instance.do_scales

    # get the spectrum
    # for MCMC runs we don't want diagnostics
    gnostics = 0
    trimspec, photspec,tauspec,cfunc = modelspec(theta,re_params,args_instance,gnostics)

    lnLik = 0.0
    wave,outspec=proc_spec(inputspec=trimspec, theta=params_instance, re_params=re_params, args_instance=args_instance, do_scales=do_scales, do_shift=do_shift)

    if args_instance.fwhm is not None:
        if (do_fudge == 1):
            s=obspec[2,::3]**2 + 10.**params_instance.tolerance_parameter_1
        else:
            s= obspec[2,::3]**2

        lnLik=-0.5*np.sum((((obspec[1,::3] - outspec[::3])**2) / s) + np.log(2.*np.pi*s))

    else:
        #Non-uniform R, the user provides the R file with flags for the number and location of the tolerance and scale parameters
        #the columns of the R file as follows: R, wl, tol_flag,scale_flag
        log_f_param = args_instance.logf_flag
        scales_param = args_instance.scales
        region_flags = np.unique(np.vstack((log_f_param, scales_param)).T, axis=0)#get unique values as a 2 column array [logf,scales]
        
        #for i,j in region_flags:
        for logf_flag_val, scale_flag_val in region_flags: #loop thru them, so we get each flags
            or_indices = np.where( (log_f_param == logf_flag_val) & (scales_param == scale_flag_val) ) #getting wl regions where both conditions are met

            if (do_fudge == 1) and (logf_flag_val > 0):
                tol_param_name=f'tolerance_parameter_{int(logf_flag_val)}'
                tol_param_index=params_instance._fields.index(tol_param_name)
                s_i = obspec[2, or_indices]**2 + 10.**params_instance[tol_param_index]
            else:
                s_i = obspec[2, or_indices]**2

            lnLik_i = -0.5 * np.sum(((obspec[1, or_indices] - outspec[or_indices])**2) / s_i + np.log(2.*np.pi*s_i))
            lnLik += lnLik_i
       
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
                    T = TPmod.set_prof(proftype,args_instance.coarsePress,press,intemp) # allow inversion 

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

    if re_params.samplemode=='mcmc':
        R2D2= params_instance.r2d2
        logg=params_instance.logg

        
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
    outspec,tmpclphotspec,tmpophotspec,cf = forwardmodel.marv(temp,logg,R2D2,args_instance.gasnames,args_instance.gasmass,logVMR,pcover,cloudmap,args_instance.cloud_opaname,cloudsize,settings.cloudata,args_instance.miewave,args_instance.mierad,cloudrad,cloudsig,cloudprof,args_instance.inlinetemps,press,args_instance.inwavenum,settings.linelist,settings.cia,args_instance.ciatemps,args_instance.use_disort,clphot,ophot,make_cf,args_instance.do_bff,bff)

    # Trim to length where it is defined.
    nwave = args_instance.inwavenum.size
    trimspec = np.zeros([2,nwave],dtype='d')
    trimspec = outspec[:,:nwave]
    cloud_phot_press = tmpclphotspec[0:npatches,:nwave].reshape(npatches,nwave)
    other_phot_press = tmpophotspec[0:npatches,:nwave].reshape(npatches,nwave)
    cfunc = np.zeros([npatches,nwave,nlayers],dtype='d')
    cfunc = cf[:npatches,:nwave,:nlayers].reshape(npatches,nwave,nlayers)
    
    trimspec[0,:] =  trimspec[0,::-1]
    trimspec[1,:] =  trimspec[1,::-1]
    
#     # now shift wavelen by delta_lambda
#     shiftspec = np.empty_like(trimspec)
# 
#     if hasattr(params_instance, "vrad"):
#         vrad = params_instance.vrad
#         dlam = trimspec[0,:] * vrad/3e5
#     else:
#         dlam = params_instance.dlambda
# 
#     shiftspec[0,:] =  trimspec[0,:] + dlam
#     shiftspec[1,:] =  trimspec[1,:]

    return trimspec, cloud_phot_press,other_phot_press,cfunc



