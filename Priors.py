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

    def __init__(self, theta, re_params,instrument_instance):
        self.re_params = re_params
        self.instrument_instance = instrument_instance
        self.args_instance = settings.runargs  # Assuming `settings.runargs` is pre-defined

        # Extract all parameters and their values
        self.all_params, self.all_params_values = utils.get_all_parametres(re_params.dictionary)
        self.params_master = namedtuple('params', self.all_params)
        self.params_instance = self.params_master(*theta)

        # Internal temperature profile keys and values
        self.intemp_keys = list(self.re_params.dictionary['pt']['params'].keys())
        self.intemp = np.array([getattr(self.params_instance, key) for key in self.intemp_keys])

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
        T = TPmod.set_prof(self.args_instance.proftype, self.args_instance.coarsePress,self.args_instance.press, self.intemp)
        prior_T = (min(T) > 1.0) and (max(T) < 6000.)


        diff=np.roll(T,-1)-2.*T+np.roll(T,1)
        pp=len(T)

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
        prior_MR = (1.0 < M < 80 and 0.5 < Rj < 2.0)


        # 4. Tolerance parameters

        priors_tolerance_params = True

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
                priors_tolerance_params = priors_tolerance_params and statement

        # priors_tolerance_params = True
        # if len(tolerance_params) > 0:
        #     for value in tolerance_params.values():
        #         statement = (
        #             0.01 * np.min(self.args_instance.obspec[2, :]**2) 
        #             < 10.**value 
        #             < 100. * np.max(self.args_instance.obspec[2, :]**2)
        #         )
        #         priors_tolerance_params = priors_tolerance_params and statement

        # Combine all priors
        post_prior = prior_T and prior_gas and prior_MR and priors_tolerance_params
        return post_prior,diff,pp
    

    def priors(self,dic):

        prior_re_params=self.get_priorranges(dic)
        prior_post,diff,pp=self.post_processing_prior()
        self.statement=(prior_re_params and prior_post)

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
            "  * Mass and Radius check: (1.0 < M < 80 and 0.5 < Rj < 2.0)\n"
            "  * Tolerance parameters: ((0.01*np.min(obspec[2,:]**2)) < 10.**tolerance_parameter < (100.*np.max(obspec[2,:]**2)))\n"
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












    




