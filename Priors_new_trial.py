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
from scipy.stats import truncnorm
from scipy.stats import norm
from copy import deepcopy


__author__ = "Fei Wang"
__copyright__ = "Copyright 2025 - Fei Wang"
__credits__ = ["Fei Wang", "Ben Burningham"]
__license__ = "GPL"
__version__ = "0.2"  
__maintainer__ = ""
__email__ = ""
__status__ = "Development"


def gaussian_prior(r, mu, sigma):

    """
    adapted from https://github.com/JohannesBuchner/MultiNest/blob/master/src/priors.f90

    Transform a uniform variable r in [0,1]
    into a Gaussian-distributed variable x.

    Parameters
    ----------
    r : float or array
        Uniform samples between 0 and 1

    mu : float
        Mean of Gaussian prior

    sigma : float
        Standard deviation of Gaussian prior

    Returns
    -------
    x : float or array
        Gaussian-distributed samples
    """

    # ppf = Percent Point Function
    #      = inverse CDF of the Gaussian
    #
    # It answers:
    # "For cumulative probability r,
    #  what Gaussian value x gives that probability?"
    #
    # Example:
    # r = 0.5  -> x = mean
    # r = 0.84 -> x ≈ mean + 1 sigma
    # r = 0.16 -> x ≈ mean - 1 sigma

    return norm.ppf(r, loc=mu, scale=sigma)


def truncated_gaussian_prior(r, mu, sigma, nleft=1, nright=2):
    """
    Transform a uniform variable r in [0,1]
    into an asymmetric truncated Gaussian-distributed variable x.

    The Gaussian is truncated to:

        [mu - nleft*sigma, mu + nright*sigma]

    Parameters
    ----------
    r : float or array
        Uniform samples between 0 and 1

    mu : float
        Mean of Gaussian prior

    sigma : float
        Standard deviation of Gaussian prior

    nleft : float
        Number of sigma below the mean

    nright : float
        Number of sigma above the mean

    Returns
    -------
    x : float or array
        Asymmetrically truncated Gaussian samples
    """

    lower = mu - nleft * sigma
    upper = mu + nright * sigma

    # Convert to standard normal space
    a = (lower - mu) / sigma   # = -nleft
    b = (upper - mu) / sigma   # = +nright

    return truncnorm.ppf(r, a, b, loc=mu, scale=sigma)
    


def uniform_prior(r, x1, x2):
    """
    Transform a uniform variable r in [0,1]
    into a uniformly distributed variable x
    between x1 and x2.

    Parameters
    ----------
    r : float or array
        Uniform samples between 0 and 1

    x1 : float
        Lower bound of the uniform prior

    x2 : float
        Upper bound of the uniform prior

    Returns
    -------
    x : float or array
        Uniformly distributed samples
        between x1 and x2
    """
    return x1 + r * (x2 - x1)


# def centered_log_abundance_prior(cube, rem):
#     """
#     Original:
#         phi = log10(rem) - cube * 12
#     """
#     return np.log10(rem) - cube * 12.


def centered_log_abundance_prior(r, factor, rem):
    """
    Notes
    -----
    Original:

        phi = log10(rem) - cube * 12

    This revised form generalizes the scaling and centering:

        phi = log10(rem) + r * factor

    allowing more flexible control of the prior width.
    """

    return np.log10(rem) + r * factor


def log_uniform_prior(r, x1, x2):
    """
    Transform a uniform variable r in [0,1]
    into a logarithmically distributed variable x
    between x1 and x2.

    Parameters
    ----------
    r : float or array
        Uniform samples between 0 and 1

    x1 : float
        Lower bound of the log prior
        (must be positive)

    x2 : float
        Upper bound of the log prior
        (must be positive)

    Returns
    -------
    x : float or array
        Logarithmically distributed samples
        between x1 and x2

    Notes
    -----
    If r <= 0, returns -1e32
    """
    if np.any(r <= 0.0):
        return -1.0e32

    lx1 = np.log10(x1)
    lx2 = np.log10(x2)

    return 10.0 ** (lx1 + r * (lx2 - lx1))



def Tp77_lndelta(r, alpha, press):
    """
    Transform a uniform variable r in [0,1]
    into a logarithmically distributed lndelta
    parameter based on the pressure grid.

    Parameters
    ----------
    r : float or array
        Uniform samples between 0 and 1

    alpha : float
        Scaling parameter

    press : array
        Pressure grid

    Returns
    -------
    lndelta : float or array
        Transformed lndelta parameter

    Notes
    -----
    If r <= 0, returns -1e32
    """

    if np.any(r <= 0.0):
        return -1.0e32

    plen = np.log10(press[-1]) - np.log10(press[0])

    pmax = alpha * plen
    p_diff = np.log(0.1) - alpha * np.log10(press[-1])

    return r * pmax + p_diff


def _get_prior_ranges(dic):
    prior_ranges = {}

    # Gas parameters
    # ---------------------------------
    gaslist = list(dic['gas'].keys())

    if gaslist[0] == 'params':

        for param in dic['gas']['params']:

            prior_ranges[param] = (
                dic['gas']['params'][param]
                .get('MC_prior_range')
            )

    else:

        for gas in gaslist:
            gas_info = dic['gas'][gas]

            # abundance
            prior_ranges[gas] = (
                gas_info['params']['log_abund']
                .get('MC_prior_range')
            )

            # p_ref
            if 'p_ref' in gas_info['params']:

                prior_ranges[f"p_ref_{gas}"] = (
                    gas_info['params']['p_ref']
                    .get('MC_prior_range')
                )

            # alpha
            if 'alpha' in gas_info['params']:

                prior_ranges[f"alpha_{gas}"] = (
                    gas_info['params']['alpha']
                    .get('MC_prior_range')
                )

    # Refinement parameters
    for param in dic['refinement_params']['params']:
        prior_ranges[param] = (
            dic['refinement_params']['params'][param]
            .get('MC_prior_range')
        )

    # PT parameters
    for param in dic['pt']['params']:

        prior_ranges[param] = (
            dic['pt']['params'][param]
            .get('MC_prior_range')
        )

    # Cloud parameters
    if 'cloud' in dic:
        # fcld
        if 'fcld' in dic['cloud']:

            prior_ranges['fcld'] = (
                dic['cloud']['fcld']
                .get('MC_prior_range')
            )

        # patch clouds
        for patch_key in dic['cloud']:

            if patch_key.startswith('patch'):

                for cloud_key in dic['cloud'][patch_key]:

                    cloud_info = dic['cloud'][patch_key][cloud_key]

                    for param in cloud_info['params']:

                        prior_ranges[param] = (
                            cloud_info['params'][param]
                            .get('MC_prior_range')
                        )

    # Added parameters
    if 'added_params' in dic:

        for param in dic['added_params']:

            prior_ranges[param] = (
                dic['added_params'][param]
                .get('MC_prior_range')
            )

    return prior_ranges


def get_all_multinest_priors(dic):
    """
    Extract Multinest_prior for every parameter.

    Returns
    -------
    prior_dict : dict
        Keys are parameter names.
        Values are corresponding Multinest_prior.
    """

    prior_dict = {}


    # Gas parameters
    gaslist = list(dic['gas'].keys())

    if gaslist[0] == 'params':
        for param in dic['gas']['params']:
            prior_dict[param] = (
                dic['gas']['params'][param]
                .get('Multinest_prior')
            )

    else:
        for gas in gaslist:
            gas_info = dic['gas'][gas]
            # abundance
            prior_dict[gas] = (
                gas_info['params']['log_abund']
                .get('Multinest_prior')
            )

            gastype = gas_info.get('gastype')
            # p_ref
            if 'p_ref' in gas_info['params']:

                prior_dict[f"p_ref_{gas}"] = (
                    gas_info['params']['p_ref']
                    .get('Multinest_prior')
                )

            # alpha
            if 'alpha' in gas_info['params']:

                prior_dict[f"alpha_{gas}"] = (
                    gas_info['params']['alpha']
                    .get('Multinest_prior')
                )

    # Refinement parameters
    for param in dic['refinement_params']['params']:
        prior_dict[param] = (
            dic['refinement_params']['params'][param]
            .get('Multinest_prior')
        )

    # PT parameters
    for param in dic['pt']['params']:
        prior_dict[param] = (
            dic['pt']['params'][param]
            .get('Multinest_prior')
        )

    # Cloud parameters
    if 'cloud' in dic:
        # fcld
        if 'fcld' in dic['cloud']:

            prior_dict['fcld'] = (
                dic['cloud']['fcld']
                .get('Multinest_prior')
            )

        # patch clouds
        for patch_key in dic['cloud']:
            if patch_key.startswith('patch'):
                for cloud_key in dic['cloud'][patch_key]:
                    cloud_info = dic['cloud'][patch_key][cloud_key]
                    for param in cloud_info['params']:
                        prior_dict[param] = (
                            cloud_info['params'][param]
                            .get('Multinest_prior')
                        )

    # Added parameters
    if 'added_params' in dic:

        for param in dic['added_params']:
            prior_dict[param] = (
                dic['added_params'][param]
                .get('Multinest_prior')
            )

    return prior_dict


PRIOR_FUNCTIONS = {
    'uniform': uniform_prior,
    'log_uniform': log_uniform_prior,
    'centered_log_abund': centered_log_abundance_prior,
    'gaussian': gaussian_prior,
    'truncated_gaussian': truncated_gaussian_prior,
    'Tp77_lndelta': Tp77_lndelta
}

def query_priors(return_dict=False):
    if return_dict:
        return PRIOR_FUNCTIONS

    return list(PRIOR_FUNCTIONS.keys())



class Priors:
    """
    A class to construct, transform, and evaluate retrieval priors for atmospheric
    parameter inference in both MCMC and MultiNest frameworks.

    This class handles:
    - User-defined prior parsing from retrieval configuration dictionaries
    - Prior transformations (MultiNest unit-cube → physical space)
    - MCMC log-prior evaluation with hard boundary and post-processing checks

    Modes
    -----
    samplemode : {'mcmc', 'multinest'}
        Controls prior evaluation strategy:
        - 'mcmc': evaluates log-prior with rejection and post-processing checks
        - 'multinest': applies unit-cube transformation to physical parameters

    Parameters
    ----------
    theta : list or ndarray
        Full parameter vector in physical space (MCMC mode).
    re_params : object
        Retrieval configuration object containing parameter definitions,
        priors, and model structure (gas, PT profile, clouds, etc.).
    args_instance : object
        Runtime configuration including pressure grid, instrument setup,
        observational data, and physical constraints (mass/radius ranges, etc.).

    Key Attributes
    --------------
    all_params : list
        Ordered list of all retrieval parameters.
    param_index : dict
        Mapping from parameter name to vector index.
    params_instance : namedtuple
        Container holding current parameter values.
    prior_dict : dict
        Raw prior specification dictionary (MultiNest mode).
    resolved_prior_dict : dict
        Dynamically updated priors with dependencies resolved during transform.
    priors : float or ndarray
        Final log-prior (MCMC) or transformed physical parameters (MultiNest).

    Methods
    -------
    transform(cube)
        Maps unit hypercube samples into physical parameter space using
        user-defined and dynamically constructed priors (MultiNest mode).
    evaluate(theta)
        Computes log-prior for MCMC sampling
    _apply_prior(r, prior_spec, *extra_args)
        Applies a named prior transformation function from PRIOR_FUNCTIONS.
    post_processing_prior()
        Enforces physical consistency after parameter construction

    """


    def __init__(self, theta, re_params, args_instance):

        self.re_params = re_params
        self.samplemode = self.re_params.samplemode.lower()

        self.args_instance = args_instance
        self.instrument_instance = args_instance.instrument

        self.Mass_priorange = args_instance.Mass_priorange
        self.R_priorange = args_instance.R_priorange

        # -------- parameters --------
        self.all_params, _ = utils.get_all_parametres(re_params.dictionary)
        self.params_master = namedtuple('params', self.all_params)
        self.param_index = {p: i for i, p in enumerate(self.all_params)}
        self.params_instance = self.params_master(*theta)

        # -------- T profile setup --------
        self.intemp_keys = list(self.re_params.dictionary['pt']['params'].keys())
        self.intemp = np.array([getattr(self.params_instance, k) for k in self.intemp_keys])

        if self.args_instance.proftype in [1, 77]:
            self.intemp = self.intemp[1:]

        # -------- gas setup --------
        self.gaslist = list(self.re_params.dictionary["gas"].keys())
        self.gastype_values = [
            info['gastype']
            for _, info in self.re_params.dictionary['gas'].items()
            if 'gastype' in info
        ]

        self.count_N = self.gastype_values.count('N')

        if self.samplemode == 'mcmc':

            # evaluate MCMC prior
            self.evaluate(theta)

        elif self.samplemode == 'multinest':


            self._build_gas_parameter_list()
            self._build_pt_parameter_list()
            self._build_cloud_parameter_list()
            self._build_refine_parameter_list()

            self.prior_dict = get_all_multinest_priors(re_params.dictionary)

            # evaluate Multinest prior
            self.transform(theta)

  
    # =========================================================
    # ------------------ MULTINEST TRANSFORM ------------------

    def transform(self, cube):
        """
        MultiNest prior transform.

        Parameters
        ----------
        cube : ndarray
            Unit cube parameters in [0,1]

        Returns
        -------
        phi : ndarray
            Physical parameters
        """
        phi = np.zeros_like(cube)

        self.resolved_prior_dict = deepcopy(self.prior_dict)

        self._transform_gas(cube, phi)
        self._transform_tp(cube, phi)
        self._transform_cloud(cube, phi)
        self._transform_refine(cube, phi)

        self.priors = phi



    # PRIOR HELPER 
    def _apply_prior(self, r, prior_spec, *extra_args):

        """
        Apply user-defined prior transform.
        """

        if prior_spec is None:
            return r

        prior_name = prior_spec[0]
        prior_args = prior_spec[1:]

        if prior_name not in PRIOR_FUNCTIONS:
            raise ValueError(
                f"Unknown prior function: {prior_name}")

        func = PRIOR_FUNCTIONS[prior_name]

        return func(r, *prior_args, *extra_args)

    # BUILD PARAM LISTS 
    def _build_gas_parameter_list(self):

        self.gaspara = []
        for i, gas in enumerate(self.gaslist):
            self.gaspara.append(gas)

            if self.gastype_values[i] == 'N':
                self.gaspara += [f"p_ref_{gas}",f"alpha_{gas}"]

            elif self.gastype_values[i] == 'H':
                self.gaspara += [f"p_ref_{gas}"]

    def _build_pt_parameter_list(self):
        self.ptpara = list(self.re_params.dictionary['pt']['params'].keys())

    def _build_cloud_parameter_list(self):
        self.cloudpara = []

        if 'cloud' not in self.re_params.dictionary:
            self.unique_cloudpara = []
            return

        cloud_dic = self.re_params.dictionary['cloud']

        # Global cloud parameters
        if 'fcld' in cloud_dic:
            self.cloudpara.append('fcld')

        # Patch-specific cloud parameters
        for patch_key, patch_val in cloud_dic.items():

            if not patch_key.startswith('patch'):
                continue
            for cloud_key, cloud_val in patch_val.items():
                params = cloud_val.get('params', [])
                self.cloudpara.extend(params)

        # Remove duplicates while preserving order
        self.unique_cloudpara = list(dict.fromkeys(self.cloudpara))

    def _build_refine_parameter_list(self):

        self.refinepara = list(
            self.re_params.dictionary['refinement_params']['params'].keys()
        )

        if 'added_params' in self.re_params.dictionary:

            self.refinepara += list(self.re_params.dictionary['added_params'].keys())


    # GAS TRANSFORM 
    def _transform_gas(self, cube, phi):
        press = self.args_instance.press
        rem = 1.0

        for name in self.gaspara:
            idx = self.param_index[name]
            prior_spec = self.resolved_prior_dict.get(name)

            # GAS ABUNDANCES 
            if name in self.gaslist:
                if prior_spec is None:
                    phi[idx] = cube[idx]
                else:
                    prior_name = prior_spec[0]
                    # dependent prior 
                    if prior_name == 'centered_log_abund':
                        phi[idx] = self._apply_prior(cube[idx],prior_spec,rem)
                    # independent prior 
                    else:
                        phi[idx] = self._apply_prior(cube[idx],prior_spec)
                rem -= 10.0 ** phi[idx]

            #p_ref
            elif name.startswith('p_ref'):
                if prior_spec is None:
                    phi[idx] = (cube[idx]* (np.log10(press[-1])- np.log10(press[0]))+ np.log10(press[0]))
                else:
                    phi[idx] = self._apply_prior(cube[idx],prior_spec)

            #alpha
            elif name.startswith('alpha'):
                if prior_spec is None:
                    phi[idx] = cube[idx]
                else:
                    phi[idx] = self._apply_prior(cube[idx],prior_spec)



    # TP TRANSFORM 
    def _transform_tp(self, cube, phi):

        press = self.args_instance.press
        pt = self.args_instance.proftype

        # DYNAMIC PRIORS
        if pt == 2:
            if self.resolved_prior_dict['logP1'] is None:
                self.resolved_prior_dict['logP1'] = ['uniform', np.log10(press[0]),np.log10(press[-1])]

            if self.resolved_prior_dict['logP3'] is None:
                self.resolved_prior_dict['logP3'] = ['uniform',phi[self.param_index['logP1']],np.log10(press[-1])]

        elif pt == 7:
            # NO INVERSION : T3 = T1 + d_T2 + d_T3,   T2 = T1 + d_T2 --> T3 > T2 > T1
            if self.resolved_prior_dict['T2'] is None:
                #T2 > T1 
                self.resolved_prior_dict['T2'] = [
                    'uniform',
                    phi[self.param_index['T1']],
                    phi[self.param_index['T1']] + 1000.
                ]
                
            if self.resolved_prior_dict['T3'] is None:
                # T3 > T2
                self.resolved_prior_dict['T3'] = [
                    'uniform',
                    phi[self.param_index['T2']],
                    phi[self.param_index['T2']] + 1000.
                ]

            if self.resolved_prior_dict['Tint'] is None:
                # Tint > T3
                self.resolved_prior_dict['Tint'] = [
                    'uniform',
                    phi[self.param_index['T3']],
                    phi[self.param_index['T3']] + 1000.
                ]

        #LOOP
        for name in self.ptpara:
            idx = self.param_index[name]
            
            if pt == 77:
                if name =='lndelta' and self.resolved_prior_dict['lndelta'] is None:
                    self.resolved_prior_dict['lndelta']=['Tp77_lndelta', phi[self.param_index['alpha']],press]

            prior_spec = self.resolved_prior_dict.get(name)

            if prior_spec is None:
                phi[idx] = cube[idx]

            else:
                phi[idx] = self._apply_prior(cube[idx],prior_spec)


    #CLOUD TRANSFORM
    def _transform_cloud(self, cube, phi):

        press = self.args_instance.press

        for name in self.unique_cloudpara:

            idx = self.param_index[name]
            prior_spec = self.resolved_prior_dict.get(name)

            #dynamic defaults
            if prior_spec is None:
                if 'logp' in name.lower():
                    prior_spec = ['uniform',np.log10(press[0]),np.log10(press[-1])]
                    self.resolved_prior_dict[name] = prior_spec
                #dp 
                elif 'dp' in name.lower():
                    related_logp = name.replace('dp', 'logp')

                    if related_logp in self.param_index:
                        logp_val = phi[self.param_index[related_logp]]
                        prior_spec = ['uniform',0,logp_val - np.log10(press[0])]
                        self.resolved_prior_dict[name] = prior_spec
  
            #apply prior 
            if prior_spec is None:
                phi[idx] = cube[idx]
            else:
                phi[idx] = self._apply_prior(cube[idx],prior_spec)


    # REFINE TRANSFORM 
    def _transform_refine(self, cube, phi):

        args = self.args_instance

        for name in self.refinepara:
            idx = self.param_index[name]
            prior_spec = self.resolved_prior_dict.get(name)

            #tolerance params 
            if name.startswith('tolerance_parameter') and prior_spec is None:
                tol_idx = int(name.split('_')[-1])
                s_indices = np.where(args.logf_flag == float(tol_idx))
                minerr = np.log10((0.01 * np.min(args.obspec[2, s_indices]))**2.)
                maxerr = np.log10((100. * np.max(args.obspec[2, s_indices]))**2.)

                phi[idx] = (cube[idx]* (maxerr - minerr)+ minerr)
                self.resolved_prior_dict[name]= ['uniform',minerr,maxerr]
                continue

            # normal params
            if prior_spec is None:
                phi[idx] = cube[idx]
            else:
                phi[idx] = self._apply_prior(cube[idx],prior_spec)




    # =========================================================
    # ------------------ MCMC prior ---------------------------

    def evaluate(self, theta):
        """MCMC log prior"""
        self.params_instance = self.params_master(*theta)

        self.prior_ranges = _get_prior_ranges(self.re_params.dictionary)
        ok_basic = self._check_param_ranges(theta, self.all_params,self.prior_ranges)
        ok_post, diff, pp, post_check_info = self.post_processing_prior()

        self.statement=(ok_basic and ok_post)
        self.post_check_info=post_check_info

        if ok_basic and ok_post:
            if self.args_instance.proftype in [1, 77]:
                gamma = self.params_instance.gamma
                logbeta = -5.0
                beta = 10.**logbeta
                alpha = 1.0

                invgamma = ((beta**alpha)/math.gamma(alpha)) * (gamma**(-alpha-1)) * np.exp(-beta/gamma)
                prprob =(-0.5/gamma)*np.sum(diff[1:-1]**2) - 0.5*pp*np.log(gamma) + np.log(invgamma)
                self.priors =prprob 
            else:
                self.priors =0.0

        else:
            self.priors = -np.inf
    


    def _check_param_ranges(self,theta, all_params,ranges):
        for param, value in zip(all_params, theta):

            r = ranges.get(param)

            if r is not None:
                if not (r[0] < value < r[1]):
                    print(param, value)
                    return False

        return True


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
            prior_T_params=(min(self.intemp) > 1.0) and (max(self.intemp) < 6000.)
            # if prior_T_params==True:
            #     T = TPmod.set_prof(self.args_instance.proftype, self.args_instance.coarsePress,self.args_instance.press, self.intemp)
            #     prior_T_overall = (min(T) > 1.0) and (max(T) < 6000.)
            diff=np.roll(self.intemp,-1)-2.*self.intemp+np.roll(self.intemp,1)
            pp=len(self.intemp)
            prior_T_overall=True

        elif self.args_instance.proftype==9:
            prior_T_params=True
            prior_T_overall=(min(self.args_instance.prof) > 1.0) and (max(self.args_instance.prof) < 6000.)

        prior_T=prior_T_overall and prior_T_params

        # 2. Gas profile check

        prior_gas=True
        if self.args_instance.chemeq==0:

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

                        if (np.log10(self.args_instance.press[0]) <= P_gas <= np.log10(self.args_instance.press[-1])):
                            gas_profile[gas_profile_index, :] = gas_nonuniform.non_uniform_gas(self.args_instance.press, P_gas, t_gas, gas_alpha)
                        else:
                            gas_profile[gas_profile_index, :] = -30
                        gas_profile_index += 1
                prior_gas = prior_gas and (np.all(gas_profile > -25.0) and np.all(gas_profile < 0.0))

        # 3. Mass and Radius check
        D = 3.086e+16 * self.args_instance.dist  # Distance in meters
        R = np.sqrt(self.params_instance.r2d2) * D if self.params_instance.r2d2 > 0. else -1.0
        g = (10.**self.params_instance.logg) / 100.
        self.M = (R**2 * g / 6.67E-11) / 1.898E27
        self.Rj = R / 69911.e3

        prior_MR = (self.Mass_priorange[0] <self.M < self.Mass_priorange[1] and self.R_priorange[0] < self.Rj < self.R_priorange[1])


        # 4. Tolerance parameters

        prior_tolerance_params = True

        tolerance_params = {
            attr: getattr(self.params_instance, attr) 
            for attr in dir(self.params_instance) if attr.startswith("tolerance_parameter")
        }

        if tolerance_params:
            if self.args_instance.fwhm is not None:
                statement = (
                    0.01 * np.min(self.args_instance.obspec[2,:]**2)
                    < 10.**list(tolerance_params.values())[0]
                    < 100. * np.max(self.args_instance.obspec[2,:]**2)
                )
                prior_tolerance_params = prior_tolerance_params and statement


            else:
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
            cloudparams = cloud_dic_new.cloud_unpack(self.re_params, self.params_instance)

            #build cloud list
            patch_numbers = [
                int(k.split(' ')[1])
                for k in self.re_params.dictionary['cloud'].keys()
                if k.startswith('patch') and k.split(' ')[1].isdigit()
            ]
            npatches = max(patch_numbers)

            cloudname_set = []
            for i in range(npatches):
                for key in self.re_params.dictionary['cloud'][f'patch {i+1}']:
                    if 'clear' not in key and key not in cloudname_set:
                        cloudname_set.append(key)

            nclouds = len(cloudname_set)

            # classify clouds 
            pattern_dis = re.compile(r'\b(deck|slab)\b', re.IGNORECASE)
            pattern_opa = re.compile(r'\b(powerlaw|grey|Mie)\b', re.IGNORECASE)

            cloud_distype = []
            cloud_opatype = []

            for name in cloudname_set:
                dis = pattern_dis.search(name)
                opa = pattern_opa.search(name)

                cloud_distype.append(dis.group(1).lower() if dis else 'unknown')
                cloud_opatype.append(opa.group(1).lower() if opa else 'unknown')


            cloud_tau0_all = np.empty(nclouds)
            cloud_top_all = np.empty(nclouds)
            cloud_bot_all = np.empty(nclouds)
            cloud_height_all = np.empty(nclouds)
            w0_all = np.empty(nclouds)
            taupow_all = np.empty(nclouds)
            loga_all = np.empty(nclouds)
            b_all = np.empty(nclouds)

            logp_bottom = np.log10(self.args_instance.press[-1])

            #helper func
            def _deck(idx):
                cloud_tau0_all[idx] = 1.0
                cloud_bot_all[idx] = logp_bottom
                cloud_top_all[idx] = cloudparams[1, idx]
                cloud_height_all[idx] = cloudparams[2, idx]

            def _slab(idx):
                cloud_tau0_all[idx] = cloudparams[0, idx]
                cloud_bot_all[idx] = cloudparams[1, idx]
                cloud_height_all[idx] = cloudparams[2, idx]
                cloud_top_all[idx] = cloud_bot_all[idx] - cloud_height_all[idx]

            def _set_defaults(idx):
                taupow_all[idx] = 0.0
                loga_all[idx] = 0.0
                b_all[idx] = 0.5


            #main loop 
            for idx, name in enumerate(cloudname_set):

                opa = cloud_opatype[idx]
                dist = cloud_distype[idx]

                #distribution
                if dist == "deck":
                    _deck(idx)
                elif dist == "slab":
                    _slab(idx)

                #opacity
                if opa == 'grey':
                    w0_all[idx] = cloudparams[3, idx]
                    _set_defaults(idx)

                elif opa == 'powerlaw':
                    w0_all[idx] = cloudparams[3, idx]
                    taupow_all[idx] = cloudparams[4, idx]
                    loga_all[idx] = 0.0
                    b_all[idx] = 0.5

                elif opa == 'mie':
                    w0_all[idx] = 0.5
                    taupow_all[idx] = 0.0
                    loga_all[idx] = cloudparams[3, idx]
                    b_all[idx] = cloudparams[4, idx]

            #cloud prior checks
            prior_cloud = (
                np.all(cloud_tau0_all >= 0.0)
                and np.all(cloud_tau0_all <= 100.0)
                and np.all(cloud_top_all < cloud_bot_all)
                and np.all(cloud_bot_all <= logp_bottom)
                and np.all(np.log10(self.args_instance.press[0]) <= cloud_top_all)
                and np.all(cloud_height_all > 0.0)
                and np.all(cloud_height_all < 7.0)
                and np.all(0.0 < w0_all)
                and np.all(w0_all <= 1.0)
                and np.all(-10.0 < taupow_all)
                and np.all(taupow_all < 10.0)
                and np.all(-3.0 < loga_all)
                and np.all(loga_all < 3.0)
                and np.all(0.0 < b_all)
                and np.all(b_all < 1.0)
            )


        # Combine all priors
        post_prior = prior_T and prior_gas and prior_MR and prior_tolerance_params and prior_cloud

        post_check_info = (
            f"prior_T: {prior_T}, "
            f"prior_gas: {prior_gas}, "
            f"prior_MR: ({prior_MR}, mass={self.M:.1f}, Radius={self.Rj:.1f}), "
            f"prior_tolerance_params: {prior_tolerance_params}, "
        )

        if self.re_params.dictionary['cloud']:
            post_check_info += f"prior_cloud: {prior_cloud}"
        else:
            post_check_info += "prior_cloud: None"

        return post_prior,diff,pp,post_check_info
    




    def __str__(self):
        """
        Provides a summary of all MCMC priors considered, including parameter priors with their ranges.

        Returns
        -------
        str
            String representation of priors.
        """

        if self.samplemode == 'mcmc':
            # Combine all_params and their prior ranges
            param_prior_list = [f"  * {param}: {self.prior_ranges.get(param, 'No prior range defined')}"
                for param in self.all_params]
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
        
        elif self.samplemode == 'multinest':

            param_prior_list = []

            for p in self.all_params:
                prior= self.resolved_prior_dict.get(p)

                if prior is None:
                    prior_text = "Default transform"
                elif prior[0]=='Tp77_lndelta':
                    prior_text = prior[:-1]+['press']
                else:
                    prior_text = str(prior)

                param_prior_list.append(
                    f"  * {p}: {prior_text}"
                )

            param_prior_text = "\n".join(param_prior_list)

            return (
                "All priors considered in the retrieval:\n"
                "------------\n"
                "- MultiNest parameter transforms:\n"
                f"{param_prior_text}\n"
            )
                












