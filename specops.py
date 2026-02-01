import numpy as np
from bbconv import prism
from bbconv import convfwhm
from bbconv import convr
import utils
from collections import namedtuple
from rotBroadInt import rot_int_cmj as rotBroad   

def proc_spec(inputspec,theta,re_params, args_instance, do_scales=True,do_shift=True):

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

    Returns
    -------
    wave_obs : ndarray, shape (Nobs,)
        Observed wavelength grid (copied from args_instance.obspec).

    outspec : ndarray, shape (Nobs,)
        Processed model spectrum on the observed grid.
    """

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
        outspec= prism_non_uniform(args_instance.obspec,modspec,args_instance.fwhm)
    else:
        # Non-uniform resolution using user-supplied R file
        log_f_param = args_instance.logf_flag       
        scales_param = args_instance.scales
        outspec= np.zeros_like(args_instance.obspec[1,:])
        
        region_flags = np.unique(np.vstack((log_f_param, scales_param)).T, axis=0)#get unique values as a 2 column array [logf,scales]
        #for i,j in region_flags:
        for logf_flag_val, scale_flag_val in region_flags: #loop thru them, so we get each flags
            or_indices = np.where( (log_f_param == logf_flag_val) & (scales_param == scale_flag_val) ) #getting wl regions where both conditions are met
            obs_wl_i = args_instance.obspec[0, :]
            spec_i = conv_non_uniform_R(modspec[1, :], modspec[0, :], args_instance.R[or_indices], obs_wl_i[or_indices])

            if do_scales==True and scale_flag_val > 0:
                # IF THERE ARE SCALE PARAMETERS
                    scale_name = f"scale{int(scale_flag_val)}"
                    if scale_name in params_instance._fields:
                        scale_value = getattr(params_instance, scale_name)
                        spec_i = scale_value * spec_i
                
            outspec[or_indices] = spec_i 
            
    return args_instance.obspec[0,:],outspec 
   
   

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
    
    

def conv_non_uniform_R(model_flux, model_wl, R, obs_wl):
    """
    Convolve a model spectrum with a wavelength-dependent resolving power 
    onto the observed wavelength grid 

    Parameters:
    - model_flux: 1D array of model flux values.
    - model_wl: 1D array of model wl values.
    - obs_wl: 1D array of observed wl values.
    - R: 1D array of resolving power values (for the obs_wl grid.)

    Returns:
    - convolved_flux: 1D array of convolved flux values on the obs_wl grid.
    """
    # create the array for the convolved flux
    convolved_flux = np.zeros_like(obs_wl)

    for i, wl_center in enumerate(obs_wl): 
        
        # compute FWHM and sigma for each wl
        # print('wl_center', wl_center)
        # print('R[i]', R[i])
        
        fwhm = wl_center / R[i]
        # print('fwhm', fwhm)
        sigma = fwhm / 2.355


        # compute the Gaussian kernel for the current wl
       
        gaussian_kernel = np.exp(-((model_wl-wl_center) ** 2) / (2 * sigma **2))
        #print('gaussian_kernel before normalisation', gaussian_kernel)

        # normalisation
        gaussian_kernel /= np.sum(gaussian_kernel)
        # print('gaussian_kernel after normalisation', gaussian_kernel)



        # apply the kernel to the flux
        convolved_flux[i] = np.sum(model_flux * gaussian_kernel)
    
    return convolved_flux
