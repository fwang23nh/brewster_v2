#!/usr/bin/env python

""" Module of processes to interpret cloud parameters from Brewster in test_module"""
from __future__ import print_function
import TPmod
import gas_nonuniform
from collections import namedtuple
import utils
import settings
import numpy as np
import math
import cloud_dic_new
import re



__author__ = "Fei Wang"
__copyright__ = "Copyright 2025 - Fei Wang"
__credits__ = ["Fei Wang", "Ben Burningham"]
__license__ = "GPL"
__version__ = "0.2"  
__maintainer__ = ""
__email__ = ""
__status__ = "Development"



class Priors:
    """
    A class to generate user-defined priors for retrieval.

    Parameters
    ----------
    theta : list
        List of parameter values.
    re_params : object
        Object containing retrieval parameters and their configurations.

    Methods
    -------
    get_priorranges(dic)
        Recursively extracts prior ranges from a dictionary.
    get_retrieval_param_priors(all_params, params_instance, priorranges)
        Validates if the retrieval parameters fall within their defined priors.
    post_processing_prior()
        Validates post-retrieval priors such as T-profile, gas profile, mass-radius, and tolerance parameters.
    """

    def __init__(self, theta, re_params,instrument_instance,Mass_priorange=[1.0,80.0],R_priorange=[0.5,2.0]):
        self.re_params = re_params
        self.instrument_instance = instrument_instance
        self.args_instance = settings.runargs  # Assuming `settings.runargs` is pre-defined

        self.Mass_priorange= Mass_priorange
        self.R_priorange= R_priorange

        # Extract all parameters and their values
        self.all_params, self.all_params_values = utils.get_all_parametres(re_params.dictionary)
        self.params_master = namedtuple('params', self.all_params)
        self.params_instance = self.params_master(*theta)

        # Internal temperature profile keys and values
        self.intemp_keys = list(self.re_params.dictionary['pt']['params'].keys())
        self.intemp = np.array([getattr(self.params_instance, key) for key in self.intemp_keys])
        if (self.args_instance.proftype == 1 or self.args_instance.proftype == 77):
            self.intemp=self.intemp[1:]

        # Extract gas type information
        self.gastype_values = [info['gastype'] for key, info in self.re_params.dictionary['gas'].items() if 'gastype' in info]
        self.count_N = self.gastype_values.count('N')

        self.priors(re_params.dictionary)


    def get_priorranges(self, dic):
        """
        Recursively extracts prior ranges from a dictionary.

        Parameters
        ----------
        dic : dict
            Dictionary containing parameter definitions and prior ranges.

        Returns
        -------
        list
            A list of prior ranges.
        """
        priorranges = []

        def recurse(d):
            if isinstance(d, dict):
                for key, value in d.items():
                    if key == 'range':
                        priorranges.append(value)
                    else:
                        recurse(value)
            elif isinstance(d, list):
                for item in d:
                    recurse(item)

        recurse(dic)
        return priorranges

    def get_retrieval_param_priors(self, all_params, params_instance, priorranges):
        """
        Validates retrieval parameters against their defined priors.

        Parameters
        ----------
        all_params : list
            List of all parameter names.
        params_instance : namedtuple
            Instance of parameters with their values.
        priorranges : list
            List of prior ranges for each parameter.

        Returns
        -------
        bool
            True if all parameters are within their priors, False otherwise.
        """
        priors = True
        for i in range(len(all_params)):
            if priorranges[i] is not None:
                statement = (priorranges[i][0] < getattr(params_instance, all_params[i]) < priorranges[i][1])
                priors = priors and statement
        return priors

    def post_processing_prior(self):
        """
        Validates post-retrieval priors including T-profile, gas profile, mass-radius, and tolerance parameters.

        Returns
        -------
        bool
            True if all post-processing priors are satisfied, False otherwise.
        """
        # 1. T-profile check
        diff=0
        pp=0
        prior_T_params=False

        if self.args_instance.proftype==2:
            """
            (0. < a1 < 1. and 0. < a2 < 1.0
            and T3 > 0.0 and P3 >= P1 and P1 >= np.log10(press[0])
            and P3 <= 5):

            """
            prior_T_params= (0. < self.params_instance.alpha1 < 1. and 0. < self.params_instance.alpha2 < 1.0
            and self.params_instance.T3 > 0.0 and self.params_instance.logP3 >= self.params_instance.logP1 and self.params_instance.logP1 >= np.log10(self.args_instance.press[0])
            and self.params_instance.logP3 <= 5)

            prior_T_overall=False
            if prior_T_params==True:
                T = TPmod.set_prof(self.args_instance.proftype, self.args_instance.coarsePress,self.args_instance.press, self.intemp)
                prior_T_overall = (min(T) > 1.0) and (max(T) < 6000.)

    
        elif self.args_instance.proftype==3:
            """
            (0. < a1 < 1. and 0. < a2 < 1.0
            and T3 > 0.0 and P3 >= P2 and P3 >= P1 and P2 >= np.log10(press[0]) and P1 >= np.log10(press[0])
             and P3 <= 5)
            """

            prior_T_params=  (0. < self.params_instance.alpha1 < 1. and 0. < self.params_instance.alpha2 < 1.0
            and self.params_instance.T3 > 0.0 and self.params_instance.logP3 >= self.params_instance.logP2  and self.params_instance.logP3 >= self.params_instance.logP1 and self.params_instance.logP2 >= np.log10(self.args_instance.press[0]) and self.params_instance.logP1 >= np.log10(self.args_instance.press[0])
             and  self.params_instance.logP3 <= 5)
            
            prior_T_overall=False
            if prior_T_params==True:
                T = TPmod.set_prof(self.args_instance.proftype, self.args_instance.coarsePress,self.args_instance.press, self.intemp)
                prior_T_overall = (min(T) > 1.0) and (max(T) < 6000.)
            

        elif self.args_instance.proftype==7:

            """
            delta=np.exp(lndelta)
            tau=delta*(press)**alpha

            delta=tau/(press)**alpha, alpha in [1,2]

            find prior range of delta to keep τ ≈ 1 in a physically sensible region (typically around 0.1 -- 10 bar).

            1. tau=1, alpha=1:

            delta=1/press, press in [0.1, 10]
            delta in [0.1,10]

            2. tau=1, alpha=2:

            delta=1/press**2, press in [0.1, 10]
            delta in [0.01,100]
            """

            delta=np.exp(self.params_instance.lndelta)
            T = np.empty([self.args_instance.press.size])
            T[:] = -100.

            P1 = ((1/delta)**(1/self.params_instance.alpha)) # P1 - pressure where tau = 1
            cp = 0.84*14.32 + 0.16*5.19
            cv = 0.84*10.16 + 0.16*3.12
            gamma=cp/cv

            tau=delta*(self.args_instance.press)**self.params_instance.alpha
            T_edd=(((3/4)*self.params_instance.Tint**4)*((2/3)+(tau)))**(1/4)
            nabla_ad=(gamma-1)/gamma
            nabla_rad = np.diff(np.log(T_edd))/np.diff(np.log(self.args_instance.press))
            convtest = np.any(np.where(nabla_rad >= nabla_ad))
            # Now get temperatures on the adiabat from RC boundary downwards
            if convtest:
                RCbound = np.where(nabla_rad >= nabla_ad)[0][0]
                P_RC = self.args_instance.press[RCbound]
            else:
                P_RC = 1000.

            prior_T_params= (1 < self.params_instance.alpha  < 2. and P_RC < 100 and P1 < P_RC
                and P_RC > self.args_instance.press[0] and  P1 > self.args_instance.press[0]
                and self.params_instance.T1 > 0.0 and self.params_instance.T2 > 0.0 and self.params_instance.T3 > 0.0 and self.params_instance.Tint >0.0 and 0.01 <= delta <= 100)
            
            prior_T_overall=False
            if prior_T_params==True:
                T = TPmod.set_prof(self.args_instance.proftype, self.args_instance.coarsePress,self.args_instance.press, self.intemp)
                prior_T_overall = (min(T) > 1.0) and (max(T) < 6000.)

        elif self.args_instance.proftype==77:
            """
            delta=np.exp(lndelta)
            tau=delta*(press)**alpha

            delta=tau/(press)**alpha, alpha in [1,2]

            find prior range of delta to keep τ ≈ 1 in a physically sensible region (typically around 0.1 -- 10 bar).

            1. tau=1, alpha=1:

            delta=1/press, press in [0.1, 10]
            delta in [0.1,10]

            2. tau=1, alpha=2:

            delta=1/press**2, press in [0.1, 10]
            delta in [0.01,100]
        
            # bits for smoothing in prior
            gam = params_instance.gamma
            """

            delta= np.exp(self.params_instance.lndelta)
            T = np.empty([self.args_instance.press.size])
            T[:] = -100.
            P1 = ((1/delta)**(1/self.params_instance.alpha))
            # put prior on P1 to put it shallower than 100 bar   
            prior_T_params= (1 < self.params_instance.alpha  < 2. and P1 < 100. and P1 > self.args_instance.press[0]
                and self.params_instance.T1 > 0.0 and self.params_instance.T2 > 0.0 and self.params_instance.T3 > 0.0 and self.params_instance.Tint >0.0 and self.params_instance.gamma>0 and 0.01 <= delta <= 100)
            
            prior_T_overall=False
            if prior_T_params==True:
                T = TPmod.set_prof(self.args_instance.proftype, self.args_instance.coarsePress,self.args_instance.press, self.intemp)
                prior_T_overall = (min(T) > 1.0) and (max(T) < 6000.)
                diff=np.roll(T,-1)-2.*T+np.roll(T,1)
                pp=len(T)

        elif self.args_instance.proftype==1:
            T = TPmod.set_prof(self.args_instance.proftype, self.args_instance.coarsePress,self.args_instance.press, self.intemp)
            prior_T_overall = (min(T) > 1.0) and (max(T) < 6000.)
            diff=np.roll(T,-1)-2.*T+np.roll(T,1)
            pp=len(T)

        prior_T=prior_T_overall and prior_T_params

        # 2. Gas profile check
        gas_keys = list(self.re_params.dictionary['gas'].keys())
        invmr = np.array([getattr(self.params_instance, key) for key in gas_keys])
        prior_gas = (np.sum(10.**invmr) < 1.0)

        if self.count_N > 0:
            gas_profile = np.full((self.count_N, self.args_instance.press.size), -1.0)
            gas_profile_index = 0
            for i, gastype in enumerate(self.gastype_values):
                if gastype == "N":
                    P_gas = getattr(self.params_instance, f"p_ref_{gas_keys[i]}")
                    gas_alpha = getattr(self.params_instance, f"alpha_{gas_keys[i]}")
                    t_gas = getattr(self.params_instance, gas_keys[i])
                    if (0. < gas_alpha < 1. and -12.0 < t_gas < 0.0 and 
                        np.log10(self.args_instance.press[0]) <= P_gas <= 2.4):
                        gas_profile[gas_profile_index, :] = gas_nonuniform.non_uniform_gas(
                            self.args_instance.press, P_gas, t_gas, gas_alpha
                        )
                    else:
                        gas_profile[gas_profile_index, :] = -30
                    gas_profile_index += 1
            prior_gas = prior_gas and (np.all(gas_profile > -25.0) and np.all(gas_profile < 0.0))

        # 3. Mass and Radius check
        D = 3.086e+16 * self.args_instance.dist  # Distance in meters
        R = np.sqrt(self.params_instance.r2d2) * D if self.params_instance.r2d2 > 0. else -1.0
        g = (10.**self.params_instance.logg) / 100.
        M = (R**2 * g / 6.67E-11) / 1.898E27
        Rj = R / 69911.e3

        prior_MR = (self.Mass_priorange[0] < M < self.Mass_priorange[1] and self.R_priorange[0] < Rj < self.R_priorange[1])


        # 4. Tolerance parameters

        prior_tolerance_params = True

        tolerance_params = {
            attr: getattr(self.params_instance, attr) 
            for attr in dir(self.params_instance) if attr.startswith("tolerance_parameter")
        }

        if tolerance_params:

            self.logf_flag=self.instrument_instance.logf_flag
            self.wl=self.instrument_instance.wl
            # Find indices where the value changes
            change_indices = np.where(np.diff(self.logf_flag) != 0)[0] + 1

            # Add 0 at the start and len at the end to get full segment boundaries
            starts = np.insert(change_indices, 0, 0)
            ends = np.append(change_indices, len(self.logf_flag))

            # Create list of [start, end] for each block
            index_blocks = [[int(s), int(e - 1)] for s, e in zip(starts, ends)]

            for key, value in tolerance_params.items():

                # Extract integer index from parameter name (e.g., "tolerance_parameter_3" -> 2)
                idx = int(key.split("_")[-1]) - 1  # zero-based index
                mask = np.where((self.args_instance.obspec[0, :] >= self.wl[index_blocks[idx][0]]) &
                (self.args_instance.obspec[0, :] <= self.wl[index_blocks[idx][1]]))

                statement = (
                    0.01 * np.min(self.args_instance.obspec[2,mask]**2)
                    < 10.**value
                    < 100. * np.max(self.args_instance.obspec[2,mask]**2)
                )
                prior_tolerance_params = prior_tolerance_params and statement



        # 5.cloud 
        prior_cloud = True
        if self.re_params.dictionary['cloud']:

            cloudparams = cloud_dic_new.cloud_unpack(self.re_params,self.params_instance) 

            patch_numbers = [
            int(k.split(' ')[1])
            for k in list(self.re_params.dictionary['cloud'].keys())
            if k.startswith('patch') and k.split(' ')[1].isdigit()
        ]
            npatches = max(patch_numbers)

            cloudname_set = [] #Used a set (cloudname_set) to automatically remove duplicates.
            for i in range(npatches):
                for key in self.re_params.dictionary['cloud'][f'patch {i+1}']:
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

            
            cloud_tau0_all = np.empty([nclouds])
            cloud_top_all = np.empty_like(cloud_tau0_all)
            cloud_bot_all = np.empty_like(cloud_tau0_all)
            cloud_height_all  = np.empty_like(cloud_tau0_all)
            w0_all = np.empty_like(cloud_tau0_all)
            taupow_all =  np.empty_like(cloud_tau0_all)
            loga_all = np.empty_like(cloud_tau0_all)
            b_all = np.empty_like(cloud_tau0_all)

            for idx, name in enumerate(cloudname_set):
                if  cloud_opatype[idx] == 'grey':
                    if cloud_distype[idx] == "slab":
                            cloud_tau0_all[idx]= cloudparams[0,idx]
                            # cloud_top_all[idx]= cloudparams[1,idx]
                            # cloud_height_all[idx]= cloudparams[2,idx]
                            # cloud_bot_all[idx] = cloud_top_all[idx] + cloud_height_all[idx]
                            cloud_bot_all[idx]= cloudparams[1,idx]
                            cloud_height_all[idx]= cloudparams[2,idx]
                            cloud_top_all[idx] = cloud_bot_all[idx] - cloud_height_all[idx]
                            w0_all[idx] = cloudparams[3,idx]
                            taupow_all[idx]= 0.0
                            loga_all[idx] = 0.0
                            b_all[idx] = 0.5
                    elif cloud_distype[idx] == "deck":
                            cloud_tau0_all[idx] = 1.0
                            cloud_bot_all[idx] = np.log10(self.args_instance.press[-1])
                            cloud_top_all[idx] = cloudparams[1,idx]
                            cloud_height_all[idx] = cloudparams[2,idx]
                            w0_all[idx] = cloudparams[3,idx]
                            taupow_all[idx] = 0.0
                            loga_all[idx] = 0.0
                            b_all[idx] = 0.5
                elif  cloud_opatype[idx] == 'powerlaw':
                    if cloud_distype[idx] == "slab":
                            cloud_tau0_all[idx] = cloudparams[0,idx]
                            # cloud_top_all[idx]= cloudparams[1,idx]
                            # cloud_height_all[idx]= cloudparams[2,idx]
                            # cloud_bot_all[idx] = cloud_top_all[idx] + cloud_height_all[idx]
                            cloud_bot_all[idx]= cloudparams[1,idx]
                            cloud_height_all[idx]= cloudparams[2,idx]
                            cloud_top_all[idx] = cloud_bot_all[idx] - cloud_height_all[idx]
                            w0_all[idx] = cloudparams[3,idx]
                            taupow_all[idx] = cloudparams[4,idx]
                            loga_all[idx] = 0.0
                            b_all[idx] = 0.5
                    elif cloud_distype[idx] == "deck":
                            cloud_tau0_all[idx] = 1.0
                            cloud_bot_all[idx] = np.log10(self.args_instance.press[-1])
                            cloud_top_all[idx] = cloudparams[1,idx]
                            cloud_height_all[idx] = cloudparams[2,idx]
                            w0_all[idx] = cloudparams[3,idx]
                            taupow_all[idx] = cloudparams[4,idx]
                            loga_all[idx] = 0.0
                            b_all[idx] = 0.5
                elif  cloud_opatype[idx] == 'mie':
                    if cloud_distype[idx] == "slab":
                        cloud_tau0_all[idx] =  cloudparams[0,idx]
                        # cloud_top_all[idx]= cloudparams[1,idx]
                        # cloud_height_all[idx]= cloudparams[2,idx]
                        # cloud_bot_all[idx] = cloud_top_all[idx] + cloud_height_all[idx]
                        cloud_bot_all[idx]= cloudparams[1,idx]
                        cloud_height_all[idx]= cloudparams[2,idx]
                        cloud_top_all[idx] = cloud_bot_all[idx] - cloud_height_all[idx]
                        w0_all[idx] = 0.5
                        taupow_all[idx] = 0.0
                        loga_all[idx] = cloudparams[3,idx]
                        b_all[idx] = cloudparams[4,idx]
                        
                    elif cloud_distype[idx] == "deck":
                        cloud_tau0_all[idx] = 1.0
                        cloud_bot_all[idx] = np.log10(self.args_instance.press[self.args_instance.press.size-1])
                        cloud_top_all[idx] = cloudparams[1,idx]
                        cloud_height_all[idx] = cloudparams[2,idx]
                        w0_all[idx] = +0.5
                        taupow_all[idx] =0.0
                        loga_all[idx] =  cloudparams[3,idx]
                        b_all[idx] =  cloudparams[4,idx]

            prior_cloud  = prior_cloud and ((np.all(cloud_tau0_all >= 0.0))
                        and (np.all(cloud_tau0_all <= 100.0))
                        and np.all(cloud_top_all < cloud_bot_all)
                        and np.all(cloud_bot_all <= np.log10(self.args_instance.press[-1]))
                        and np.all(np.log10(self.args_instance.press[0]) <= cloud_top_all)
                        and np.all(cloud_top_all < cloud_bot_all)
                        and np.all(0. < cloud_height_all)
                        and np.all(cloud_height_all < 7.0)
                        and np.all(0.0 < w0_all)
                        and np.all(w0_all <= 1.0)
                        and np.all(-10.0 < taupow_all)
                        and np.all(taupow_all < +10.0)
                        and np.all( -3.0 < loga_all)
                        and np.all (loga_all < 3.0)
                        and np.all(b_all < 1.0)
                        and np.all(b_all > 0.0))


        # Combine all priors
        post_prior = prior_T and prior_gas and prior_MR and prior_tolerance_params and prior_cloud

        post_check_info = (
            f"prior_T: {prior_T}, "
            f"prior_gas: {prior_gas}, "
            f"prior_MR: {prior_MR}, "
            f"prior_tolerance_params: {prior_tolerance_params}, "
        )

        if self.re_params.dictionary['cloud']:
            post_check_info += f"prior_cloud: {prior_cloud}"
        else:
            post_check_info += "prior_cloud: None"

        return post_prior,diff,pp,post_check_info
    

    def priors(self,dic):

        re_params_priorranges=self.get_priorranges(dic)
        prior_re_params=self.get_retrieval_param_priors(self.all_params,self.params_instance,re_params_priorranges)
        prior_post,diff,pp,post_check_info=self.post_processing_prior()
        self.statement=(prior_re_params and prior_post)

        self.post_check_info=post_check_info

        if self.statement == True:
            if self.args_instance.proftype == 1 or self.args_instance.proftype==77:
                logbeta = -5.0
                beta=10.**logbeta
                alpha=1.0
                x=self.params_instance.gamma
                invgamma=((beta**alpha)/math.gamma(alpha)) * (x**(-alpha-1)) * np.exp(-beta/x)
                prprob = (-0.5/self.params_instance.gamma)*np.sum(diff[1:-1]**2) - 0.5*pp*np.log(self.params_instance.gamma) + np.log(invgamma)

                self.priors =prprob 
            else:
                self.priors =0.0

        else:
            self.priors = -np.inf


    def __str__(self):
        """
        Provides a summary of all priors considered, including parameter priors with their ranges.

        Returns
        -------
        str
            String representation of priors.
        """
        # Combine all_params and their prior ranges
        param_prior_list = [
            f"  * {param}: {priorrange if priorrange else 'No prior range defined'}"
            for param, priorrange in zip(self.all_params, self.get_priorranges(self.re_params.dictionary))
        ]
        param_prior_text = "\n".join(param_prior_list)

        return (
            "All priors considered in the retrieval:\n"
            "------------\n"
            "- Parameter Priors: Defined by user ranges\n"
            f"{param_prior_text}\n"
            "--------------------\n"
            "- Post-processing Priors:\n"
            "  * T-profile check: (min(T) > 1.0) and (max(T) < 6000.)\n" 
            "  * Gas profile check: (np.sum(10.**(invmr)) < 1.0) and valid gas profiles\n"
            # "  * Mass and Radius check: (1.0 < M < 80 and 0.5 < Rj < 2.0)\n"
            "  * Mass and Radius check:\n"
            f"({self.Mass_priorange[0]} < M < {self.Mass_priorange[1]}) and "
            f"({self.R_priorange[0]} < Rj < {self.R_priorange[1]})\n"
            "  * Tolerance parameters: ((0.01*np.min(obspec[2,:]**2)) < 10.**tolerance_parameter < (100.*np.max(obspec[2,:]**2)))\n"
            "  *all cloud parameters should be within range\n"
            "  * Prior check results:\n"
            f"{self.post_check_info}\n"
        )



# 1. T_profile check (min(T) > 1.0) and (max(T) < 6000.) 
    

#     intemp_keys = list(re_params.dictionary['pt']['params'].keys())
#     intemp = np.array([getattr(params_instance, key) for key in intemp_keys])
#     T = TPmod.set_prof(proftype,junkP,press,intemp)



# 2. gas_profile check


#      (np.sum(10.**(invmr)) < 1.0),  np.all(gas_profile > -25.0) and np.all(gas_profile < 0.0)
    
#     gastype_values = [info['gastype'] for key, info in re_params.dictionary['gas'].items() if 'gastype' in info]
#     count_N = gastype_values.count('N')

#     if count_N>0:
#     gas_profile = np.full((count_N, press.size), -1.0)
#     gas_profile_index =0
#     for i in range(len(gastype_values)):
#         if  gastype_values[i]=="N":
#             P_gas= getattr(params_instance, "p_ref_%s"%gas_keys[i])
#             gas_alpha= getattr(params_instance, "alpha_%s"%gas_keys[i])
#             t_gas= getattr(params_instance, gas_keys[i])
#             if (0. < gas_alpha < 1. and -12.0 < t_gas < 0.0  and np.log10(press[0]) <= P_gas <= 2.4):
#                 gas_profile[gas_profile_index,:]=gas_nonuniform.non_uniform_gas(press,P_gas,t_gas,gas_alpha)
#             else:
#                 gas_profile[gas_profile_index,:]=-30
#             gas_profile_index+=1

#         if  gastype_values[i]=="H":
#                 P_hgas= getattr(params_instance, "p_ref_%s"%gas_keys[i])

# 3. Mass and R prior  

#         and  1.0 < M < 80  and  0.5 < Rj < 2.0

#         D = 3.086e+16 * dist
#         R = -1.0
#         if (r2d2 > 0.):
#             R = np.sqrt(r2d2) * D
#         g = (10.**logg)/100.
#         M = (R**2 * g/(6.67E-11))/1.898E27
#         Rj = R / 69911.e3


# 4. tolerance_parameter_1   


#  ((0.01*np.min(obspec[2,:]**2)) < 10.**tolerance_parameter < (100.*np.max(obspec[2,:]**2)))



def priormap_dic(theta,re_params):

    all_params,all_params_values =utils.get_all_parametres(re_params.dictionary) 
    params_master = namedtuple('params',all_params)
    params_instance = params_master(*theta)

    args_instance=settings.runargs

    # Unpack all necessary parameters into local variables
    press=args_instance.press
    fwhm=args_instance.fwhm
    obspec=args_instance.obspec
    proftype=args_instance.proftype
    do_fudge=args_instance.do_fudge

    phi = np.zeros_like(theta)
    gaslist=list(re_params.dictionary["gas"].keys())
    gastype_values = [info['gastype'] for key, info in re_params.dictionary['gas'].items() if 'gastype' in info]

    gaspara=[]
    for i in range(len(gaslist)):
        gaspara.append(gaslist[i])
        if  gastype_values[i]=='N':
            gaspara.append("p_ref_%s"%gaslist[i])
            gaspara.append("alpha_%s"%gaslist[i])

        if  gastype_values[i]=='H':
            gaspara.append("p_ref_%s"%gaslist[i])
             
    ng=len(gaspara)
    

    if ng==2:
        phi[0] = (theta[0] * (args_instance.metscale[-1] - args_instance.metscale[0])) + args_instance.metscale[0]
        phi[1] = (theta[1] * (args_instance.coscale[-1] -  args_instance.coscale[0])) +  args_instance.coscale[0]
        
    else:
        rem = 1
        for i in range(0, ng):
            if gaspara[i] in gaslist:
                phi[i] = np.log10(rem) -  (theta[i] * 12.)
                rem = rem - (10**phi[i])
            elif gaspara[i].startswith('p_ref'):
                phi[i]= (theta[i]* \
                             (np.log10(press[-1]) - np.log10(press[0]))) + np.log10(press[0])

            elif gaspara[i].startswith('alpha'):
                phi[i]= theta[i]

    max_mass = 80. # jupiters
    min_mass = 1.0 # jupiters
    min_rad = 0.5 # jupiters
    max_rad = 2.5 # jupiters
    
    
    mass_index=params_instance._fields.index('M')
    
    # this is a simple uniform prior on mass
    # we want to use the radius, to set a mass prior. 
    # this will correlate these parameters??? Yes. which is fine.
    phi[mass_index] = (theta[mass_index] * (max_mass - min_mass)) + min_mass

    # this is if we want log g prior: phi[ng] = theta[ng] * 5.5
    # now we retrieve radius in R_jup 
    R_index=params_instance._fields.index('R')
    R_j = ((max_rad - min_rad)*theta[R_index]) + min_rad
    phi[R_index] = R_j
    
    if (fwhm == 555):
        log_f_param = args_instance.logf_flag
        log_f_param_max = int(np.max(log_f_param))
        
        scales_param = args_instance.scales
        nonzero_scales = sorted(set(scales_param) - {0})  

        if nonzero_scales:  
            for i in nonzero_scales:
                pname = f"scale{i}"
                p_index = params_instance._fields.index(pname)
                phi[p_index] = (theta[p_index] * 1.5) + 0.5
 
        # now dlam
        dlam_index=params_instance._fields.index('dlambda')
        phi[dlam_index] = (theta[dlam_index] * 0.02) - 0.01

        if (do_fudge == 1):
            for i in range(1, log_f_param_max + 1):
                s_indices = np.where(log_f_param == float(i))
                minerr = np.log10((0.01 * np.min(obspec[2, s_indices]))**2.)
                maxerr = np.log10((100 * np.max(obspec[2, s_indices]))**2.)

                tol_param_name=f'tolerance_parameter_{i}'
                tol_param_index=params_instance._fields.index(tol_param_name)
                phi[tol_param_index] = (theta[tol_param_index] * (maxerr-minerr)) + minerr

       # if (do_fudge == 1):
       #     minerr1 = np.log10((0.01 * np.min(obspec[2, s2]))**2.)
       #     maxerr1 = np.log10((100 * np.max(obspec[2, s2]))**2.)
       #     tolerance_parameter_1_index = params_instance._fields.index('tolerance_parameter_1')
       #     phi[tolerance_parameter_1_index] = (theta[tolerance_parameter_1_index] * (maxerr1 - minerr1)) + minerr1


       #     if s3[0].size > 0:
       #         minerr2 = np.log10((0.01 * np.min(obspec[2, s3]))**2.)
       #         maxerr2 = np.log10((100 * np.max(obspec[2, s3]))**2.)
       #         tolerance_parameter_2_index = params_instance._fields.index('tolerance_parameter_2')
       #         phi[tolerance_parameter_2_index] = (theta[tolerance_parameter_2_index] * (maxerr2 - minerr2)) + minerr2

        
    elif (fwhm < 0.0):
        if (fwhm == -1 or fwhm == -3 or fwhm == -4 or fwhm == -8):
            s1  = np.where(obspec[0,:] < 2.5)
            s2  = np.where(np.logical_and(obspec[0,:] > 2.5,obspec[0,:] < 5.0))
            s3 =  np.where(obspec[0,:] > 5.)
            # scale parameters here - generous factor 2 either side??
            scale1_index=params_instance._fields.index('scale1')
            scale2_index=params_instance._fields.index('scale2')
            phi[scale1_index] = (theta[scale1_index] * 1.5) + 0.5
            phi[scale2_index] = (theta[scale2_index] * 1.5) + 0.5
            # now dlam
            dlam_index=params_instance._fields.index('dlambda')
            phi[dlam_index] = (theta[dlam_index] * 0.02) - 0.01
            if (do_fudge == 1):
                # These are tolerances for each band due to difference SNRs
                minerr_s1 =np.log10((0.01 *  np.min(obspec[2,s1]))**2.)
                maxerr_s1 =np.log10((100.*np.max(obspec[2,s1]))**2.)
                tolerance_parameter_1_index=params_instance._fields.index('tolerance_parameter_1')
                phi[tolerance_parameter_1_index] = (theta[tolerance_parameter_1_index] * (maxerr_s1 - minerr_s1)) + minerr_s1
                
                minerr_s2 =np.log10((0.01 *  np.min(obspec[2,s2]))**2.)
                maxerr_s2 =np.log10((100.*np.max(obspec[2,s2]))**2.)
                tolerance_parameter_2_index=params_instance._fields.index('tolerance_parameter_2')
                phi[tolerance_parameter_2_index] = (theta[tolerance_parameter_2_index] * (maxerr_s2 - minerr_s2)) + minerr_s2
                
                minerr_s3 =np.log10((0.01 *  np.min(obspec[2,s3]))**2.)
                maxerr_s3 = np.log10((100.*np.max(obspec[2,s3]))**2.)
                tolerance_parameter_3_index=params_instance._fields.index('tolerance_parameter_3')
                phi[tolerance_parameter_3_index] = (theta[tolerance_parameter_3_index] * (maxerr_s3 - minerr_s3)) + minerr_s3

        elif (fwhm == -2):
            s1  = np.where(obspec[0,:] < 2.5)
            s3 =  np.where(obspec[0,:] > 5.)
            # scale parameter
            scale1_index=params_instance._fields.index('scale1')
            phi[scale1_index] = (theta[scale1_index] * 1.5) + 0.5
            # dlam now:
            dlam_index=params_instance._fields.index('dlambda')
            phi[dlam_index] = (theta[dlam_index] * 0.02) - 0.01
            if (do_fudge == 1):
                # These are tolerances for each band due to difference SNR
                minerr_s1 = np.log10((0.01 *  np.min(obspec[2,s1]))**2.)
                maxerr_s1 = np.log10((100.*np.max(obspec[2,s1]))**2.)
                tolerance_parameter_1_index=params_instance._fields.index('tolerance_parameter_1')
                phi[tolerance_parameter_1_index] = (theta[tolerance_parameter_1_index] * (maxerr_s1 - minerr_s1)) + minerr_s1
                
                minerr_s3 = np.log10((0.01 *  np.min(obspec[2,s3]))**2.)
                maxerr_s3 = np.log10((100.*np.max(obspec[2,s3]))**2.)
                tolerance_parameter_2_index=params_instance._fields.index('tolerance_parameter_2')
                phi[tolerance_parameter_2_index] = (theta[tolerance_parameter_2_index] * (maxerr_s3 - minerr_s3)) + minerr_s3

        elif (fwhm == -6):
            ##Geballe NIR CGS4 data
            s1  = np.where(obspec[0,:] < 1.585)
            s3  = np.where(obspec[0,:] > 1.585)
            #not including relative scale factor since data is calibrated to the same photometry
            #dlam:
            dlam_index=params_instance._fields.index('dlambda')
            phi[dlam_index] = (theta[dlam_index] * 0.02) - 0.01
            #Tolerance parameter (only one):
            if (do_fudge==1):
                minerr = np.log10((0.01 *  np.min(obspec[2,:]))**2.)
                maxerr = np.log10((100.*np.max(obspec[2,:]))**2.)
                tolerance_parameter_1_index=params_instance._fields.index('tolerance_parameter_1')
                phi[tolerance_parameter_1_index] = (theta[tolerance_parameter_1_index] * (maxerr - minerr)) + minerr

        elif (fwhm == -7): #Geballe NIR + NIRC + CGS4 MIR
            s1 = np.where(obspec[0, :] < 1.585)
            s2 = np.where(obspec[0, :] > 1.585)
            s3 = np.where(np.logical_and(obspec[0, :] > 2.52, obspec[0, :] < 4.2))  #NIRC
            s4 = np.where(obspec[0, :] > 4.2) #CGS4
            # scale parameter
            scale1_index=params_instance._fields.index('scale1')
            scale2_index=params_instance._fields.index('scale2')
            phi[scale1_index] = (theta[scale1_index] * 1.5) + 0.5
            phi[scale2_index] = (theta[scale2_index] * 1.5) + 0.5
            #dlam
            dlam_index=params_instance._fields.index('dlambda')
            phi[dlam_index] = (theta[dlam_index] * 0.02) - 0.01
            if (do_fudge == 1):
                # These are tolerances for each band due to difference SNRs
                minerr_s1 =np.log10((0.01 *  np.min(obspec[2,s1]))**2.)
                maxerr_s1 =np.log10((100.*np.max(obspec[2,s1]))**2.)
                tolerance_parameter_1_index=params_instance._fields.index('tolerance_parameter_1')
                phi[tolerance_parameter_1_index] = (theta[tolerance_parameter_1_index] * (maxerr_s1 - minerr_s1)) + minerr_s1
                
                minerr_s2 =np.log10((0.01 *  np.min(obspec[2,s2]))**2.)
                maxerr_s2 =np.log10((100.*np.max(obspec[2,s2]))**2.)
                tolerance_parameter_2_index=params_instance._fields.index('tolerance_parameter_2')
                phi[tolerance_parameter_2_index] = (theta[tolerance_parameter_2_index] * (maxerr_s2 - minerr_s2)) + minerr_s2
                
                minerr_s3 =np.log10((0.01 *  np.min(obspec[2,s3]))**2.)
                maxerr_s3 = np.log10((100.*np.max(obspec[2,s3]))**2.)
                tolerance_parameter_3_index=params_instance._fields.index('tolerance_parameter_3')
                phi[tolerance_parameter_3_index] = (theta[tolerance_parameter_3_index] * (maxerr_s3 - minerr_s3)) + minerr_s3


    else:
        # this just copes with normal, single instrument data
        # so do dlam next
        dlam_index=params_instance._fields.index('dlambda')
        phi[dlam_index] = (theta[dlam_index] * 0.02) - 0.01
        # now fudge
        if (do_fudge == 1):
            # logf here
            minerr =np.log10((0.01 *  np.min(obspec[2,:]))**2.)
            maxerr = np.log10((100.*np.max(obspec[2,:]))**2.)
            tolerance_parameter_1_index=params_instance._fields.index('tolerance_parameter_1')
            phi[tolerance_parameter_1_index] = (theta[tolerance_parameter_1_index] * (maxerr - minerr)) + minerr

 
    if (proftype == 1):

        intemp_keys = list(re_params.dictionary['pt']['params'].keys())
        gam_index=params_instance._fields.index(intemp_keys[0])   
        phi[gam_index] = theta[gam_index] *5000

        tempkeys=intemp_keys[1:]
        for i in range(len(tempkeys)):
            index=params_instance._fields.index(tempkeys[i])                                            
            phi[index] = theta[index] *3999  + 1

    if (proftype == 2):
                   
        alpha1_index=params_instance._fields.index('alpha1')
        alpha2_index=params_instance._fields.index('alpha2')
        logP1_index=params_instance._fields.index('logP1')
        logP3_index=params_instance._fields.index('logP3')
        T3_index=params_instance._fields.index('T3')
                                                                  
        # a1
        phi[alpha1_index] = 0.25 + (theta[alpha1_index]*0.25)
        # a2
        phi[alpha2_index] = 0.1 + (theta[alpha2_index] * 0.1)
        #P1
        phi[logP1_index] = (theta[logP1_index]* \
                             (np.log10(press[-1]) - np.log10(press[0]))) + np.log10(press[0])
        #P3
        #P3 must be greater than P1
        phi[logP3_index] = (theta[logP3_index] * \
                             (np.log10(press[-1]) - phi[logP1_index])) + phi[logP1_index]
        #T3
        phi[T3_index] = (theta[T3_index] * 3000.) + 1500.0


    elif (proftype == 3):
                                               
                                               
        alpha1_index=params_instance._fields.index('alpha1')
        alpha2_index=params_instance._fields.index('alpha2')
        logP1_index=params_instance._fields.index('logP1')
        logP2_index=params_instance._fields.index('logP2')
        logP3_index=params_instance._fields.index('logP3')
        T3_index=params_instance._fields.index('T3')
                                               

        # a1
        phi[alpha1_index] = 0.25 + (theta[alpha1_index]*0.25)
        # a2
        phi[alpha2_index] = 0.1 * (theta[alpha2_index] * 0.1)
        #P3 in press[0]--press[-1]
        phi[logP3_index] = (theta[logP3_index] * (np.log10(press[-1]) - np.log10(press[0]))) + np.log10(press[0])                                        
                                               
        # press[0]<P1<P3
        phi[logP1_index] = (theta[logP1_index]* (phi[logP3_index] - np.log10(press[0]))) + np.log10(press[0])
        ## press[0]<P2<P3
        phi[logP2_index] = (theta[logP2_index]* (phi[logP3_index] - np.log10(press[0]))) + np.log10(press[0])
       
        #T3
        phi[T3_index] = (theta[T3_index] * 3000.) + 1500.

    
    elif (proftype == 7):
                                                   
        Tint_index=params_instance._fields.index('Tint')
        alpha_index=params_instance._fields.index('alpha')
        lndelta_index=params_instance._fields.index('lndelta')
        T1_index=params_instance._fields.index('T1')
        T2_index=params_instance._fields.index('T2')
        T3_index=params_instance._fields.index('T3')
                                                                   
                                               
       # Tint - prior following Molliere+2020
      #  phi[Tint_index] = 300 + (theta[Tint_index] * 2000)  #UNCOMMENT IN CASE OF INVERSION
        # alpha, between 1 and 2
        phi[alpha_index] = theta[alpha_index] + 1. 
        # lndlelta
        plen = np.log10(press[-1]) - np.log10(press[0])
        pmax=phi[alpha_index]*plen 
        p_diff=np.log(0.1)-phi[alpha_index]*np.log10(press[-1])
        phi[lndelta_index] = theta[lndelta_index]*pmax+p_diff                                           

        # T1
       # phi[T1_index] = 10. + (theta[T1_index] * 4000)
        # T2
       # phi[T2_index] = 10. + (theta[T2_index] * 4000)
        # T3
       # phi[T3_index] = 10.+ (theta[T3_index] * 4000)

        # IN CASE OF NO INVERSION UNCOMMENT THESE: T3 = T1 + d_T2 + d_T3,   T2 = T1 + d_T2 --> T3 > T2 > T1

        phi[T1_index] = 10. + (theta[T1_index] *4000)

       #T2 > T1 
        delta_T2 = theta[T2_index] * 1000
        phi[T2_index] = phi[T1_index] + delta_T2

       # T3 > T2 
        delta_T3 = theta[T3_index] * 1000
        phi[T3_index] = phi[T2_index] + delta_T3

       # Tint > T3
        delta_Tint = theta[Tint_index] * 1000
        phi[Tint_index] = phi[T3_index] + delta_Tint

    elif (proftype == 77):
                                                   
        Tint_index=params_instance._fields.index('Tint')
        alpha_index=params_instance._fields.index('alpha')
        lndelta_index=params_instance._fields.index('lndelta')
        T1_index=params_instance._fields.index('T1')
        T2_index=params_instance._fields.index('T2')
        T3_index=params_instance._fields.index('T3')
                                                                   
                                               
       # Tint - prior following Molliere+2020
        phi[Tint_index] = 300 + (theta[Tint_index] * 2000)
        # alpha, between 1 and 2
        phi[alpha_index] = theta[alpha_index] + 1. 
        # lndlelta
        plen = np.log10(press[-1]) - np.log10(press[0])
        pmax=phi[alpha_index]*plen 
        p_diff=np.log(0.1)-phi[alpha_index]*np.log10(press[-1])
        phi[lndelta_index] = theta[lndelta_index]*pmax+p_diff                                           

        # T1
        phi[T1_index] = 10. + (theta[T1_index] * 4000)
        # T2
        phi[T2_index] = 10. + (theta[T2_index] * 4000)
        # T3
        phi[T3_index] = 10.+ (theta[T3_index] * 4000)

        intemp_keys = list(re_params.dictionary['pt']['params'].keys())
        gam_index=params_instance._fields.index(intemp_keys[0])   
        phi[gam_index] = theta[gam_index] *5000


    npatches = args_instance.cloudmap.shape[0]
    # only really ready for 2 patches here
    if (npatches > 1):
        fcld_index=params_instance._fields.index('fcld')
        phi[fcld_index] = theta[fcld_index]

    

    if np.all(args_instance.cloudmap!= 0):
        cloudlist=[]
        for i in range(1, npatches+1):
            for key in re_params.dictionary['cloud']['patch %s' % i].keys():
                if 'clear' not in key:
                    cloudlist.append(key)

        for cloud in cloudlist:
            if cloud=='grey cloud deck':
            # 'cloudnum': 99,'cloudtype':2,
                logp_gcd_index=params_instance._fields.index('logp_gcd')
                dp_gcd_index=params_instance._fields.index('dp_gcd')
                #cloud top
                phi[logp_gcd_index] = \
                    (theta[logp_gcd_index] *(np.log10(press[-1]) \
                                    - np.log10(press[0])))\
                                    + np.log10(press[0])
                # cloud height
                phi[dp_gcd_index] = theta[dp_gcd_index] * 7.
                        
            elif cloud=='grey cloud slab':
            # 'cloudnum': 99,'cloudtype':1,
                tau_gcs_index=params_instance._fields.index('tau_gcs')
                logp_gcs_index=params_instance._fields.index('logp_gcs')
                dp_gcs_index=params_instance._fields.index('dp_gcs')
                # cloud tau
                phi[tau_gcs_index] = theta[tau_gcs_index]*100.
                #cloud base
                phi[logp_gcs_index] = \
                    (theta[logp_gcs_index] *(np.log10(press[-1]) \
                                        - np.log10(press[0]))) \
                                        + np.log10(press[0])
                # cloud height
                phi[dp_gcs_index] = theta[dp_gcs_index] *\
                    (phi[logp_gcs_index] - np.log10(press[0]))
                                    
        
            elif cloud=='powerlaw cloud deck':
            # 'cloudnum': 89,'cloudtype':2,
                logp_pcd_index=params_instance._fields.index('logp_pcd')
                dp_pcd_index=params_instance._fields.index('dp_pcd')
                alpha_pcd_index=params_instance._fields.index('alpha_pcd') 
                #cloud top
                phi[logp_pcd_index] = \
                            (theta[logp_pcd_index] *(np.log10(press[-1]) \
                                            - np.log10(press[0]))) \
                                            + np.log10(press[0])
                # cloud height
                phi[dp_pcd_index] = theta[dp_pcd_index] * 7.
                # power law
                phi[alpha_pcd_index] = (theta[alpha_pcd_index] * 20.) - 10.


            elif 'Mie scattering cloud deck' in cloud:
            #   'cloudnum': cloudnum,'cloudtype':2,

                cloudspecies=cloud.split('--')[1].strip()
                logp_pcd_index=params_instance._fields.index('logp_mcd_%s'%cloudspecies)
                dp_pcd_index=params_instance._fields.index('dp_mcd_%s'%cloudspecies)
                #cloud base
                phi[logp_pcd_index] = \
                    (theta[logp_pcd_index] *(np.log10(press[-1]) \
                                    - np.log10(press[0])))\
                                    + np.log10(press[0])

                
                # cloud height
                phi[dp_pcd_index] = theta[dp_pcd_index] * 7.

                for patch, clouds in re_params.dictionary['cloud'].items():
                    for name, cloud_info in clouds.items():
                        if cloudspecies in name:
                            particle_dis = cloud_info.get('particle_dis', None)
   
                if  particle_dis=="hansen": 
                    hansen_a_mcd_index=params_instance._fields.index('hansen_a_mcd_%s'%cloudspecies)
                    hansen_b_mcd_index=params_instance._fields.index('hansen_b_mcd_%s'%cloudspecies)                                               
                    # particle effective radius
                    phi[hansen_a_mcd_index] = (theta[hansen_a_mcd_index] * 6.) - 3.
                    # particle spread
                    phi[hansen_b_mcd_index] = theta[hansen_b_mcd_index]
                elif particle_dis=="log_normal":
                    mu_mcd_index=params_instance._fields.index('mu_mcd_%s'%cloudspecies)
                    sigma_mcd_index=params_instance._fields.index('sigma_mcd_%s'%cloudspecies)
                    # particle effective radius
                    phi[mu_mcd_index] = (theta[mu_mcd_index] * 6.) - 3.
                    # particle spread
                    phi[mu_mcd_index] = theta[mu_mcd_index]                                                 

            elif cloud=='power law cloud slab':
                    # 'cloudnum': 89, 'cloudtype':1,
                    tau_pcs_index=params_instance._fields.index('tau_pcs')
                    logp_pcs_index=params_instance._fields.index('logp_pcs')
                    dp_pcs_index=params_instance._fields.index('dp_pcs')
                    alpha_pcs_index=params_instance._fields.index('alpha_pcs')

                    # cloud tau
                    phi[tau_pcs_index] = theta[tau_pcs_index]*100.
                    #cloud base
                    phi[logp_pcs_index] = \
                        (theta[logp_pcs_index]*\
                            (np.log10(press[-1]) - np.log10(press[0]))) \
                            + np.log10(press[0])
                    # cloud height
                    phi[dp_pcs_index] = \
                        theta[dp_pcs_index] * (phi[logp_pcs_index] \
                                            - np.log10(press[0]))
                    # power law
                    phi[alpha_pcs_index] = (theta[alpha_pcs_index] * 20.) - 10.


            elif 'Mie scattering cloud slab' in cloud:

                cloudspecies=cloud.split('--')[1].strip()

                tau_mcs_index=params_instance._fields.index('tau_mcs_%s'%cloudspecies)
                logp_mcs_index=params_instance._fields.index('logp_mcs_%s'%cloudspecies)
                dp_mcs_index=params_instance._fields.index('dp_mcs_%s'%cloudspecies)
                # cloud tau
                phi[tau_mcs_index] = theta[tau_mcs_index]*100.
                #cloud base
                phi[logp_mcs_index] = \
                    (theta[logp_mcs_index] *(np.log10(press[-1]) \
                                        - np.log10(press[0]))) \
                                        + np.log10(press[0])
                # cloud height
                phi[dp_mcs_index] = theta[dp_mcs_index] * \
                    (phi[logp_mcs_index] - np.log10(press[0]))

                for patch, clouds in re_params.dictionary['cloud'].items():
                    for name, cloud_info in clouds.items():
                        if cloudspecies in name:
                            particle_dis = cloud_info.get('particle_dis', None)
                                                                        
                if particle_dis=="hansen": 
                    hansen_a_mcs_index=params_instance._fields.index('hansen_a_mcs_%s'%cloudspecies)
                    hansen_b_mcs_index=params_instance._fields.index('hansen_b_mcs_%s'%cloudspecies)                                               
                    # particle effective radius
                    phi[hansen_a_mcs_index] = (theta[hansen_a_mcs_index] * 6.) - 3.
                    # particle spread
                    phi[hansen_b_mcs_index] = theta[hansen_b_mcs_index]
                elif particle_dis=="log_normal":
                    mu_mcs_index=params_instance._fields.index('mu_mcs_%s'%cloudspecies)
                    sigma_mcs_index=params_instance._fields.index('sigma_mcs_%s'%cloudspecies)
                    # particle effective radius
                    phi[mu_mcs_index] = (theta[mu_mcs_index] * 6.) - 3.
                    # particle spread
                    phi[mu_mcs_index] = theta[mu_mcs_index]


    return phi
                                                                     




    




