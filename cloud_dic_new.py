#!/usr/bin/env python

""" Module of processes to interpret cloud parameters from Brewster in test_module"""
from __future__ import print_function
import numpy as np
import scipy as sp
from scipy import interpolate
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian1DKernel
import re


__author__ = "Fei Wang"
__copyright__ = "Copyright 2024 - Fei Wang"
__credits__ = ["Fei Wang", "Ben Burningham"]
__license__ = "GPL"
__version__ = "0.2"  
__maintainer__ = ""
__email__ = ""
__status__ = "Development"



# cloudparams = np.ones([5,nclouds],dtype='d')
# cloudrad = np.zeros((nlayers,ncloud),dtype='d')
# cloudsig = np.zeros_like(cloudrad)
# cloudprof = np.zeros_like(cloudrad)
# cloudmap = np.zeros((npatches, nclouds), dtype=bool)


def cloud_unpack(re_params, params_instance):
    #CLOUDFREE
    if (not hasattr(re_params, 'dictionary')) or ('cloud' not in re_params.dictionary) or (not re_params.dictionary['cloud']):
        return np.ones((5, 0), dtype='d') * 0.0

    patch_numbers = [
        int(k.split(' ')[1])
        for k in list(re_params.dictionary['cloud'].keys())
        if k.startswith('patch') and len(k.split(' ')) > 1 and k.split(' ')[1].isdigit()
    ]
    

    # NO PATCH
    #if not patch_numbers:
    #    return np.ones((5, 0), dtype='d') * 0.0
        
    npatches = max(patch_numbers)

    # names of all clouds
    cloudname_set = []
    for patch_key in re_params.dictionary['cloud']:
        for key in re_params.dictionary['cloud'][patch_key]:
            if 'clear' not in key and key not in cloudname_set:
                cloudname_set.append(key)

    #NO CLOUD   
    nclouds = len(cloudname_set)
    if nclouds == 0:
        return np.ones((5, 0), dtype='d') * 0.0

    
    pattern_dis = re.compile(r'\b(deck|slab)\b', re.IGNORECASE)
    cloud_distype = [
        (pattern_dis.search(name).group(1).lower() if pattern_dis.search(name) else 'unknown')
        for name in cloudname_set
    ]

    pattern_opa = re.compile(r'\b(powerlaw|grey|Mie)\b', re.IGNORECASE)
    cloud_opatype = [
        (pattern_opa.search(name).group(1).lower() if pattern_opa.search(name) else 'unknown')
        for name in cloudname_set
    ]

    # defaults
    cloudparams = np.ones([5, nclouds], dtype='d')
    cloudparams[0, :] = 0.0
    cloudparams[1, :] = 0.0
    cloudparams[2, :] = 0.1
    cloudparams[3, :] = 0.0
    cloudparams[4, :] = 0.5


    
    cloudmap = np.zeros((npatches, nclouds), dtype=bool)
    for icloud, patches in enumerate(re_params.cloudpatch_index):
        for ipatch in patches:
            cloudmap[ipatch - 1, icloud] = True

    for idx, cloud in enumerate(cloudname_set):
        for i in range(npatches):
            if cloud in list(re_params.dictionary['cloud'][f'patch {i+1}'].keys()):
                cloud_rawparams_key = list(re_params.dictionary['cloud'][f'patch {i+1}'][cloud]['params'].keys())
                cloud_rawparams=np.array([getattr(params_instance, key) for key in cloud_rawparams_key])
                # cloudmap[i,idx]= 1

        if ((cloud_distype[idx] == "deck") and (cloud_opatype[idx] == 'grey')):
            cloudparams[1:4,idx] = cloud_rawparams[:]
            cloudparams[4,idx] = 0.0
        elif ((cloud_distype[idx] == "slab") and (cloud_opatype[idx] == 'grey')):
            cloudparams[0:4,idx] = cloud_rawparams[:]
            cloudparams[4,idx] = 0.0
        elif (cloud_distype[idx] == "deck") and (cloud_opatype[idx] == 'powerlaw' or cloud_opatype[idx]== 'mie'):
            cloudparams[1:5,idx] = cloud_rawparams[:]
        else:
            cloudparams[:,idx] = cloud_rawparams[:]



    return cloudparams
    
    
    
    '''
    for idx, cloud in enumerate(cloudname_set):
        # pick first patch for this cloud
        patches = re_params.cloudpatch_index[idx]
        patch_key = f'patch {patches[0]}'
        cloud_rawparams_key = list(re_params.dictionary['cloud'][patch_key][cloud]['params'].keys())
        cloud_rawparams = np.array([getattr(params_instance, key) for key in cloud_rawparams_key])

        if (cloud_distype[idx] == "deck") and (cloud_opatype[idx] == 'grey'):
            cloudparams[1:4, idx] = cloud_rawparams
            cloudparams[4, idx] = 0.0
        elif (cloud_distype[idx] == "slab") and (cloud_opatype[idx] == 'grey'):
            cloudparams[0:4, idx] = cloud_rawparams
            cloudparams[4, idx] = 0.0
        elif (cloud_distype[idx] == "deck") and (cloud_opatype[idx] in ('powerlaw', 'mie')):
            if len(cloud_rawparams) == 5:
                cloudparams[:, idx] = cloud_rawparams
            else:
                cloudparams[1:5, idx] = cloud_rawparams
        else:
            cloudparams[:, idx] = cloud_rawparams

    return cloudparams#, cloudmap
    '''

    
    
    
    '''
    for icloud, cloud in enumerate(cloudname_set):
        patches = re_params.cloudpatch_index[icloud]  # take the patches for this cloud
        patch_key = f'patch {patches[0]}'
        cloud_rawparams_key = list(re_params.dictionary['cloud'][patch_key][cloud]['params'].keys())
        cloud_rawparams = np.array([getattr(params_instance,key) for key in cloud_rawparams_key])

        if (cloud_distype[icloud]=="deck") and (cloud_opatype[icloud]=='grey'):
            cloudparams[1:4,icloud] = cloud_rawparams
            cloudparams[4,icloud] = 0.
        elif (cloud_distype[icloud]=="slab") and (cloud_opatype[icloud]=='grey'):
            cloudparams[0:4,icloud] = cloud_rawparams
            cloudparams[4,icloud] = 0.
        elif (cloud_distype[icloud]=="deck") and (cloud_opatype[icloud] in ('powerlaw','mie')):
            if len(cloud_rawparams)==5:
                cloudparams[:,icloud] = cloud_rawparams
            else:
                cloudparams[1:5,icloud] = cloud_rawparams
        else:
            cloudparams[:,icloud] = cloud_rawparams


    return cloudparams#, cloudmap
    
    
    '''    
    

'''
def cloud_unpack(re_params,params_instance):

    patch_numbers = [
    int(k.split(' ')[1])
    for k in list(re_params.dictionary['cloud'].keys())
    if k.startswith('patch') and k.split(' ')[1].isdigit()
]
    npatches = max(patch_numbers)

    cloudname_set = [] #Used a set (cloudname_set) to automatically remove duplicates.
    for i in range(npatches):
        for key in re_params.dictionary['cloud'][f'patch {i+1}']:
            if 'clear' not in key and key not in cloudname_set:
                cloudname_set.append(key)

    nclouds = len(cloudname_set)


    pattern_dis = re.compile(r'\b(deck|slab)\b', re.IGNORECASE)

    cloud_distype=[]
    for idx, name in enumerate(cloudname_set, start=1):
        match = pattern_dis.search(name)
        cloud_distype.append(match.group(1).lower() if match else 'unknown')


    pattern_opa = re.compile(r'\b(powerlaw|grey|Mie)\b', re.IGNORECASE)
    cloud_opatype=[]
    for idx, name in enumerate(cloudname_set, start=1):
        match = pattern_opa.search(name)
        cloud_opatype.append(match.group(1).lower() if match else 'unknown')


    cloudparams = np.ones([5,nclouds],dtype='d')
    cloudparams[0,:] = 0.
    cloudparams[1,:] = 0.0
    cloudparams[2,:] = 0.1
    cloudparams[3,:] = 0.0
    cloudparams[4,:] = 0.5


    # cloudmap = np.zeros((npatches, nclouds), dtype=int)


    for idx, cloud in enumerate(cloudname_set):
        for i in range(npatches):
            if cloud in list(re_params.dictionary['cloud'][f'patch {i+1}'].keys()):
                cloud_rawparams_key = list(re_params.dictionary['cloud'][f'patch {i+1}'][cloud]['params'].keys())
                cloud_rawparams=np.array([getattr(params_instance, key) for key in cloud_rawparams_key])
                # cloudmap[i,idx]= 1

        if ((cloud_distype[idx] == "deck") and (cloud_opatype[idx] == 'grey')):
            cloudparams[1:4,idx] = cloud_rawparams[:]
            cloudparams[4,idx] = 0.0
        elif ((cloud_distype[idx] == "slab") and (cloud_opatype[idx] == 'grey')):
            cloudparams[0:4,idx] = cloud_rawparams[:]
            cloudparams[4,idx] = 0.0
        elif (cloud_distype[idx] == "deck") and (cloud_opatype[idx] == 'powerlaw' or cloud_opatype[idx]== 'mie'):
            cloudparams[1:5,idx] = cloud_rawparams[:]
        else:
            cloudparams[:,idx] = cloud_rawparams[:]



    return cloudparams

'''

def atlev(l0,press):
    nlayers = press.size
    if (l0 <= nlayers-2):
        pl1 = np.exp(((1.5)*np.log(press[l0])) - ((0.5)*np.log(press[l0+1])))
        pl2 = np.exp((0.5)*(np.log(press[l0] * press[l0+1])))
    else:
        pl1 = np.exp((0.5 * np.log(press[l0-1] * press[l0])))
        pl2 = press[l0]**2 / pl1

    return pl1, pl2





# now need to translate cloudparams in to cloud profile even
# if do_clouds is zero..
# 5 entries for cloudparams for simple slab model are:
# 0) dtau at 1um
# 1) top layer id (or pressure)
# 2) base ID (these are both in 64 layers)
# 3) rg
# 4) rsig
# in the case of a simple mixto cloud (i.e. cloudnum = 99 or 89) we have:
# 0) dtau (at 1um for non-grey) 
# 1) top layer ID
# 2) bottom later ID
# 3) rg = albedo
# 4) rsig = power for tau power law





def atlas(re_params,cloudparams,press):

    # Cloud types
    # 1:  slab cloud
    # 2: deep thick cloud , we only see the top
    # 3: slab with fixed thickness log dP = 0.005 (~1% height)
    # 4: deep thick cloud with fixed height log dP = 0.005
    # In both cases the cloud properties are density, rg, rsig for real clouds
    # and dtau, w0, and power law for cloudnum = 99 or 89

    nlayers = press.size

    patch_numbers = [
    int(k.split(' ')[1])
    for k in list(re_params.dictionary['cloud'].keys())
    if k.startswith('patch') and k.split(' ')[1].isdigit()
]
    
    
    if len(patch_numbers) == 0:
    # no patches → no clouds
        return (np.zeros((press.size, 0), dtype='d'),
            np.zeros((press.size, 0), dtype='d'),
            np.zeros((press.size, 0), dtype='d'))
    
    npatches = max(patch_numbers)

    cloudname_set = [] #Used a set (cloudname_set) to automatically remove duplicates.
    for i in range(npatches):
        for key in re_params.dictionary['cloud'][f'patch {i+1}']:
            if 'clear' not in key and key not in cloudname_set:
                cloudname_set.append(key)

    nclouds = len(cloudname_set)


    pattern_dis = re.compile(r'\b(deck|slab)\b', re.IGNORECASE)

    cloud_distype=[]
    for idx, name in enumerate(cloudname_set, start=1):
        match = pattern_dis.search(name)
        cloud_distype.append(match.group(1).lower() if match else 'unknown')


    pattern_opa = re.compile(r'\b(powerlaw|grey|Mie)\b', re.IGNORECASE)
    cloud_opatype=[]
    for idx, name in enumerate(cloudname_set, start=1):
        match = pattern_opa.search(name)
        cloud_opatype.append(match.group(1).lower() if match else 'unknown')



    cloudrad = np.zeros((nlayers,nclouds),dtype='d')
    cloudsig = np.zeros_like(cloudrad)
    cloudprof = np.zeros_like(cloudrad)

    for j in range(0,nclouds):
        if (cloud_distype[j] == "slab"):
            # 5 entries for cloudparams are:
            # 0) total tau for cloud at 1 micron
            # 1) log top pressure
            # 2) pressure thickness in dex
            # 3) rg
            # 4) rsig

            tau = cloudparams[0,j]
            p2 = 10.**cloudparams[1,j]
            dP = cloudparams[2,j]
     
            p1 = p2 / 10.**dP
            rad = cloudparams[3,j]
            sig = cloudparams[4,j]
            pdiff = np.empty(nlayers,dtype='f')

            pdiff = abs(np.log(press) - np.log(p1))
            l1 = np.argmin(pdiff)
            # Whichever layer-mean-P the cloud top is closest
            # to is the +1 layer of the cloud
            # same for the -1 layer of the base
            pdiff = abs(np.log(press) - np.log(p2))
            l2 = np.argmin(pdiff)

            # This is a slab cloud
            # dtau/dP propto P
            if (l1 == l2):
                cloudprof[l1,j] = tau
            else:
                const = tau / (p2**2 - p1**2)
                # partial top fill
                pl1, pl2 = atlev(l1,press)
                cloudprof[l1,j] = const * (pl2**2 - p1**2)
                # partial bottom fill
                pl1, pl2 = atlev(l2,press)
                cloudprof[l2,j] = const *  (p2**2 - pl1**2) 
                for k in range (l1+1,l2):
                    l1,l2 = atlev(k,press)
                    cloudprof[k,j] = const * (l2**2 - l1**2)

            # We're sampling particle radius in log space        
            if (cloud_opatype[j]  == 'mie'):
                cloudrad[:,j] = 10.**rad
            else:
                cloudrad[:,j] = rad
            cloudsig[:,j] = sig        

        if (cloud_distype[j]  == "deck"):

            # 5 entries for cloudparams are:
            # 0) empty
            # 1) top pressure
            # 2) scale height (in dex)
            # 3) rg
            # 4) rsig
    
            p0 = 10.**cloudparams[1,j]
            dP = cloudparams[2,j]
  
            scale = ((p0 * 10.**dP) - p0)  / 10.**dP
            rad = cloudparams[3,j]
            sig = cloudparams[4,j]
    
            pdiff = np.empty(nlayers,dtype='f')
    
        
            # In cloud 99/89 case rsig is power law for tau~lambda^alpha 
            # Here P0 is the pressure where tau= 1 for the cloud
            # so dtau / dP = const * exp((P-P0) / scale)
            # See notes for derivation of constant and integral
            const = 1. / (1 - np.exp(-p0 / scale))
            for k in range (0,nlayers):
                pl1, pl2 = atlev(k,press)
                # now get dtau for each layer, where tau = 1 at P0
                term1 = (pl2 - p0) / scale
                term2 = (pl1 - p0) / scale
                if (term1 > 10 or term2 > 10):
                    cloudprof[k,j] = 100.00
                else:
                    cloudprof[k,j] = const * (np.exp(term1) -
                                                np.exp(term2))

            # We're sampling particle radius in log space        
            if (cloud_opatype[j] == 'mie'): 
                cloudrad[:,j] = 10.**rad
            else:
                cloudrad[:,j] = rad
            cloudsig[:,j] = sig       

        if (cloud_distype[j] not in ['slab','deck']):
            print ("cloud layout not recognised. stopping") 

    
    return cloudprof,cloudrad,cloudsig






