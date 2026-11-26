import numpy as np
from scipy.stats import binned_statistic
from forward_model.build.bbconv import prism
from forward_model.build.bbconv import convfwhm
from forward_model.build.bbconv import convr
from collections import namedtuple
from rotBroadInt import rot_int_cmj as rotBroad   

def proc_spec(inputspec,theta,re_params, args_instance, do_scales=True,do_shift=True,do_conv=True):

    """
    Apply wavelength shifts, rotational broadening, instrumental convolution,
    and optional scaling to a model spectrum.

    This function takes a high-resolution model spectrum and maps it into the
    observed space by:
      1. Applying wavelength shifts (delta-lambda and/or radial velocity)
      2. Applying rotational broadening (vsini)
      3. Convolving with the instrumental line-spread function
      4. Applying multiplicative scale factors

    Parameters
    ----------
    inputspec : ndarray, shape (2, N)
        Input model spectrum with:
        - inputspec[0, :] = wavelength array
        - inputspec[1, :] = flux array

    theta : array_like
        Current parameter vector.

    re_params : object
        Retrieval parameter container defining parameter ordering.

    args_instance : object
        configuration containing observational spectrum, resolution
        information, and region flags.

    do_scales : bool, optional
        If True, apply multiplicative scale parameters (default: True).

    do_shift : bool, optional
        If True, apply wavelength shifts and rotational broadening (default: True).

    do_conv : bool, optional
        If True, apply Instrumental convolution on modelspec (default: True).

    Returns
    -------
    wave_obs : ndarray, shape (Nobs,)
        Observed wavelength grid (copied from args_instance.obspec).

    outspec : ndarray, shape (Nobs,)
        Processed model spectrum on the observed grid.
    """

    import utils

    all_params,all_params_values =utils.get_all_parametres(re_params.dictionary)
    params_master = namedtuple('params',all_params)
    params_instance = params_master(*theta)

    modspec = None 

    # ------------------------------------------------------------------
    # Step 1: Apply wavelength shifts and rotational broadening
    # ------------------------------------------------------------------
    if do_shift == True:

        # --- Constant wavelength offset (delta-lambda) ---
        if hasattr(params_instance, "dlambda"):
            dlam = params_instance.dlambda
            modspec = np.empty_like(inputspec)
            modspec[0,:] =  inputspec[0,:] + dlam
            modspec[1,:] =  inputspec[1,:]

        # --- Radial velocity Doppler shift ---
        if hasattr(params_instance, "vrad"):
            vrad = params_instance.vrad
            dlam = inputspec[0,:] * vrad/3e5
            
            if modspec is not None:
                modspec[0,:] = modspec[0,:] + dlam
            else:
                modspec = np.empty_like(inputspec)
                modspec[0,:] = inputspec[0,:] + dlam
                modspec[1,:] = inputspec[1,:]

        # --- Rotational broadening ---
        if hasattr(params_instance, "vsini"):
            vsini = params_instance.vsini
            rotspec = rotBroad(modspec[0],modspec[1],vsini)
            modspec[1,:] = rotspec
    

    else:
        # No shifting or broadening requested
        modspec=inputspec

    # ------------------------------------------------------------------
    # Step 2: Instrumental convolution and regional scaling
    # ------------------------------------------------------------------

    if args_instance.fwhm is not None:
        # Use convolution for Spex
        if do_conv == True:
            outspec= prism_non_uniform(args_instance.obspec,modspec,args_instance.fwhm)
            wave=args_instance.obspec[0,:]
        else:
            outspec= modspec[1,:]
            wave=modspec[0,:]


    else:
        if do_conv == True:
            wave=args_instance.obspec[0,:]
            # Non-uniform resolution using user-supplied R file
            log_f_param = args_instance.logf_flag       
            scales_param = args_instance.scales
            outspec= np.zeros_like(args_instance.obspec[1,:])
            
            region_flags = np.unique(np.vstack((log_f_param, scales_param)).T, axis=0)#get unique values as a 2 column array [logf,scales]
            #for i,j in region_flags:
            for logf_flag_val, scale_flag_val in region_flags: #loop thru them, so we get each flags
                or_indices = np.where( (log_f_param == logf_flag_val) & (scales_param == scale_flag_val) ) #getting wl regions where both conditions are met
                obs_wl_i = args_instance.obspec[0, :]

                R_i = args_instance.R[or_indices]
                wl_i = obs_wl_i[or_indices]

                if getattr(args_instance, 'conv_mode', None) is None:
                    # Four-column R files contain spectroscopic resolving powers.
                    spec_i = conv_non_uniform_R(modspec[1, :], modspec[0, :], R_i, wl_i)
                else:
                    conv_value = args_instance.conv_mode[or_indices]
                    spec_i = np.zeros_like(wl_i)
                    spectroscopy = conv_value == 0
                    if np.any(spectroscopy):
                        spec_i[spectroscopy] = conv_non_uniform_R(
                            modspec[1, :], modspec[0, :], R_i[spectroscopy], wl_i[spectroscopy]
                        )
                    for j in np.flatnonzero(conv_value == 1):
                        spec_i[j] = conv_binning_values(
                            modspec[1, :], modspec[0, :], R_i[j:j+1], wl_i[j:j+1]
                        )[0]

                if do_scales==True and scale_flag_val > 0:
                    # IF THERE ARE SCALE PARAMETERS
                        scale_name = f"scale{int(scale_flag_val)}"
                        if scale_name in params_instance._fields:
                            scale_value = getattr(params_instance, scale_name)
                            spec_i = scale_value * spec_i
                    
                outspec[or_indices] = spec_i 
            
        else:
            outspec= modspec[1,:]
            wave=modspec[0,:]

    return wave,outspec 
   
   

#**************************************************************************

# This hides the fortran convolution code so it works nicely with rest of
# code

#**************************************************************************

def prism_non_uniform(obspec,modspec,resel):

    fluxout = prism(np.asfortranarray(obspec),np.asfortranarray(modspec),resel)[0:obspec[0,:].size]

    return fluxout


def conv_uniform_FWHM(obspec,modspec,fwhm):

    fluxout = convfwhm(np.asfortranarray(obspec),np.asfortranarray(modspec),fwhm)[0:obspec[0,:].size]
    
    return fluxout


        

def conv_uniform_R(obspec,modspec,R):

    fluxout = convr(np.asfortranarray(obspec),np.asfortranarray(modspec),R)[0:obspec[0,:].size]
    
    return fluxout
    
def conv_binning_values(model_flux, model_wl, BW, obs_wl):
    """
    Convolve a model spectrum with a wavelength-dependent band width onto the observed wavelength grid

    Parameters:
    - model_flux: 1D array of model flux values
    - model_wl: 1D array of model wl values
    - obs_wl: 1D array of observed center wl values
    - BW: 1D array of band-width values (for the obs_wl grid)

    Returns: 
    - convolved_flux: 1D array of convolved flux values on the obs_wl grid
    """

    # create the array for the convolved flux

    convolved_flux = np.zeros_like(obs_wl)

    # Calculate the upper and lower bounds, while also removing NaN values
    lw = (obs_wl - BW/2)
    up = (obs_wl + BW/2)
    
    lw1 = lw[~np.isnan(lw)]
    up1 = up[~np.isnan(up)]
    centers = obs_wl[~np.isnan(obs_wl)]

    # Creating bins 
    bin_edges = np.concatenate(([lw1[0]], up1))

    # Calculate Mean Flux for each channel
    convolved_flux, _, _ = binned_statistic(model_wl, model_flux, statistic='mean', bins=bin_edges)

    return convolved_flux


def _gaussian_average(model_flux, model_wl, center, sigma):
    weights = np.exp(-((model_wl - center) ** 2) / (2 * sigma ** 2))
    total = np.sum(weights)
    weights /= total
    return np.sum(model_flux * weights), total


def conv_non_uniform_R(model_flux, model_wl, R, obs_wl):
    """Convolve a model spectrum onto observed wavelengths at variable resolving power."""
    model_flux = np.asarray(model_flux)
    model_wl = np.asarray(model_wl)
    R = np.asarray(R)
    obs_wl = np.asarray(obs_wl)
    convolved_flux = np.zeros_like(obs_wl)
    window_bounds = None

    # A slightly wider window than the 8-sigma profiling prototype gives
    # additional headroom for spectra with deep features and for sparse grids.
    sigma_limit = 10.0
    if (model_wl.ndim == 1 and model_wl.size and model_flux.shape == model_wl.shape
            and obs_wl.ndim == 1 and np.issubdtype(obs_wl.dtype, np.floating)
            and R.shape == obs_wl.shape
            and np.all(np.isfinite(model_wl)) and np.all(model_wl[1:] >= model_wl[:-1])
            and np.all(np.isfinite(model_flux))
            and np.all(np.isfinite(obs_wl)) and np.all(obs_wl > 0)
            and np.all(np.isfinite(R)) and np.all(R > 0)):
        sigmas = (obs_wl / R) / 2.355
        lower = np.searchsorted(model_wl, obs_wl - sigma_limit * sigmas, side="left")
        upper = np.searchsorted(model_wl, obs_wl + sigma_limit * sigmas, side="right")
        window_bounds = lower, upper
        peak_flux = np.max(np.abs(model_flux.astype(np.result_type(model_flux, float), copy=False)))
        tail_weight = np.exp(-0.5 * sigma_limit ** 2)

    for i, center in enumerate(obs_wl):
        sigma = (center / R[i]) / 2.355
        if window_bounds is not None:
            first, last = window_bounds[0][i], window_bounds[1][i]
            if 0 < last - first < model_wl.size:
                window = slice(first, last)
                flux, weight = _gaussian_average(model_flux[window], model_wl[window], center, sigma)
                # Bound the change from discarded samples, including the change
                # in normalization. Fall back on sparse grids or when far-away
                # high flux or cancellation can make the tails important.
                omitted = (model_wl.size - (last - first)) * tail_weight
                if (peak_flux + abs(flux)) * omitted <= 1e-12 * abs(flux) * weight:
                    convolved_flux[i] = flux
                    continue
        convolved_flux[i], _ = _gaussian_average(model_flux, model_wl, center, sigma)

    return convolved_flux
