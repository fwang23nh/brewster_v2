# This script tests the forward model in the context of cloudless T8 dwarf
# parameters a drawn from a retrieval on G570D
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mgimg
import matplotlib.colors as colors
import scipy as sp
import numpy as np
import emcee
import corner
import pickle as pickle
import forwardmodel
import ciamod
import TPmod
import brewtools
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian1DKernel
from scipy.io.idl import readsav
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from bensconv import prism_non_uniform
from bensconv import conv_uniform_R
from bensconv import conv_uniform_FWHM
from collections import namedtuple
import utils
import settings
import gas_nonuniform
import test_module
from specops import proc_spec





def NoCloud_Tdwarf(xpath,xlist):
    
     fwhm=3.3
     wavelength_range=[1,2.5]
     R_file=None
     obspec = []

     ##gas
     chemeq=0
     gaslist = ['h2o','ch4','co','co2','nh3','h2s','K','Na']
     gastype_list=['U','U','U','U','U','U','U','U']
     ptype=9

     ## clouds

     do_clouds=0
     npatches=1
     cloud_name = ['clear']
     cloud_type = ['none']
     cloudpatch_index=[[1]]
     particle_dis=[]
     cloudpath=None

     # ModelConfig:
     do_fudge=0
     samplemode='mcmc'

     instrument_instance = utils.Instrument(wavelength_range=wavelength_range, R_file=R_file,obspec=obspec,fwhm=fwhm)
     re_params = utils.Retrieval_params(samplemode,chemeq,gaslist,gastype_list,do_fudge,ptype,do_clouds,npatches,cloud_name,cloud_type,cloudpatch_index,particle_dis,instrument_instance,fwhm=fwhm)
     model_config_instance = utils.ModelConfig(samplemode,do_fudge,cloudpath=cloudpath)
     io_config_instance = utils.IOConfig()



     model_config_instance.do_bff=0
     model_config_instance.malk=0
     model_config_instance.pfile='data/test_data/G570D_model_benchmark_PROFILE.dat'
     model_config_instance.xlist=xlist #'gaslistR10K.dat'
     model_config_instance.xpath=xpath
     model_config_instance.update_dictionary()


     # obspec = []#np.asfortranarray(np.loadtxt("LSR1835_data_realcalib_new_trimmed.dat",dtype='d',unpack='true')) # obs is not actually using

     args_instance = utils.ArgsGen(re_params,model_config_instance,instrument_instance,obspec)
     settings.init(args_instance)
     settings.cia = args_instance.cia
     settings.linelist= utils.get_opacities(args_instance.gaslist,args_instance.w1,args_instance.w2,args_instance.press,args_instance.xpath,args_instance.xlist,args_instance.malk)
     settings.cloudata = args_instance.cloudata
     args_instance=settings.runargs

     all_params,all_params_values =utils.get_all_parametres(re_params.dictionary) 
     params_master = namedtuple('params',all_params)
     theta=[-3.27,-3.36,-7.27,-8.28,-4.73,-8.71,-5.36]+[4.89]+[1.50901046e-19]+[0.00258329]
     params_instance = params_master(*theta)

     gnostics=0
     trimspec, cloud_phot_press,other_phot_press,cfunc=test_module.modelspec(params_instance,re_params,args_instance,gnostics)
     benchspec = np.loadtxt('data/test_data/No_cloud_800K_model_benchmark_SPEC.dat',skiprows=3,unpack=True)
     args_instance.obspec=benchspec
     wav,outspec=proc_spec(inputspec=trimspec, theta=params_instance, re_params=re_params, args_instance=args_instance, do_scales=True, do_shift=True)


     difference_spectrum = outspec / benchspec[1,:]
     print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
     print('------------------------------------------------------------')
     print(' Test for T dwarf case, forward model only, No clouds')
     print('------------------------------------------------------')
     print('mean value of modelspectrum / T benchmark = '+str(np.mean(difference_spectrum)))
     print('std deviation of modelspectrum / T benchmark = '+str(np.std(difference_spectrum)))

     percent_change = np.mean(abs(outspec - benchspec[1,:])/benchspec[1,:])
     
     if (percent_change < 0.01):
          print("less than 1percent difference with T dwarf regime benchmark")
          print('-------------------------------------------------------')
          print('   ')
          return True
     else:
          print("greater than 1 percent difference T dwarf regime with benchmark")
          print('-------------------------------------------------------')
          print('   ')
          return False




def MieClouds_Ldwarf(xpath,xlist,cloudpath):
     
     Rfile = 'examples/example_data/code_test_R_file.txt'
     obspec = np.loadtxt('data/test_data/Mie_cloud_1800K_model_benchmark_SPEC.dat',skiprows=0,unpack=True)
     wavelength_range=[1,15]
     ndata=1

     #retrieval_params
     ##gas
     chemeq=0
     gaslist = ['h2o','co','co2','ch4','tio','vo','crh','feh','k','na']
     gastype_list=['U','U','U','U','U','U','U','U','U','U']
     ptype=2

     ## clouds
     do_clouds=1
     npatches=1

     cloud_name=['MgSiO3.mieff','Fe.mieff']
     cloud_type=['slab','deck']
     cloudpatch_index=[[1],[1]]
     particle_dis=['hansen','hansen']
     cloudpath=cloudpath
     # ModelConfig:

     do_fudge = 1
     samplemode='mcmc'
     instrument_instance = utils.Instrument(wavelength_range=wavelength_range, R_file=Rfile,obspec=obspec)

     rfile = np.loadtxt(Rfile)
     instrument_instance.scales = rfile[:, 3]
     instrument_instance.logf_flag = rfile[:,2]

     re_params = utils.Retrieval_params(samplemode=samplemode, chemeq=chemeq, gaslist=gaslist,
     gastype_list=gastype_list,do_fudge=do_fudge, ptype=ptype, do_clouds=do_clouds,
     npatches=npatches, cloud_name=cloud_name, cloud_type=cloud_type,
     cloudpatch_index=cloudpatch_index, particle_dis=particle_dis,
     instrument=instrument_instance,vrad=False,vsini=False,fwhm=None)


     model_config_instance = utils.ModelConfig(samplemode,do_fudge,cloudpath=cloudpath)
     io_config_instance = utils.IOConfig()


     model_config_instance.do_bff=1
     model_config_instance.malk=0
     model_config_instance.pfile="t1700g1000f3.dat"
     model_config_instance.xlist=xlist #'gaslistR10K.dat'
     model_config_instance.xpath=xpath

     model_config_instance.dist=11.35
     model_config_instance.update_dictionary()


     args_instance = utils.ArgsGen(re_params,model_config_instance,instrument_instance,obspec)
     settings.init(args_instance)
     settings.cia = args_instance.cia
     settings.linelist= utils.get_opacities(args_instance.gaslist,args_instance.w1,args_instance.w2,args_instance.press,args_instance.xpath,args_instance.xlist,args_instance.malk)
     settings.cloudata = args_instance.cloudata


     all_params,all_params_values =utils.get_all_parametres(re_params.dictionary) 
     params_master = namedtuple('params',all_params)


     params_dict = dict(
     h2o=-3.55278369,
     co=-2.83012757,
     co2=-4.31062021,
     ch4=-4.95190596,
     tio=-9.77059307,
     vo=-8.85409603,
     crh=-8.4153943,
     feh=-7.85745521,
     K_Na=-6.5725089,
     logg=5.46814691,
     r2d2=2.68655361e-20,
     dlambda=0.00283604013,
     scale1=1.07209671,
     scale2=1.11922607,
     tolerance_parameter_1=-31.6119701,
     tolerance_parameter_2=-33.2775232,
     tolerance_parameter_3=-34.6762823,
     alpha1=0.345025551,
     alpha2=0.0678307874,
     logP1=0.0756891116,
     logP3=1.71616709,
     T3=4886.46433,
     tau_mcs_MgSiO3Cry=5.42024548,
     logp_mcs_MgSiO3Cry=-2.76574938,
     dp_mcs_MgSiO3Cry=0.438059949,
     hansan_a_mcs_MgSiO3Cry=-0.573919866,
     hansan_b_mcs_MgSiO3Cry=0.0858329576,
     logp_mcd_Fe2O3_WS15=0.872374998,
     dp_mcd_Fe2O3_WS15=4.3939299,
     hansan_a_mcd_Fe2O3_WS15=-1.96757779,
     hansan_b_mcd_Fe2O3_WS15=0.0624967679
     )

     theta = list(params_dict.values())
     params_instance = params_master(*theta)
     # print(params_instance)


     gnostics=0
     trimspec, cloud_phot_press,other_phot_press,cfunc=test_module.modelspec(params_instance,re_params,args_instance,gnostics)
     wave,topspec=proc_spec(inputspec=trimspec, theta=params_instance, re_params=re_params, args_instance=args_instance, do_scales=args_instance.do_scales, do_shift=args_instance.do_shift)

     obspec=args_instance.obspec
     outspec = topspec
     difference_spectrum = outspec / obspec[1,:]

     print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
     print('------------------------------------------------------------')
     print(' Test for L dwarf case, plotting via theta and testmodule')
     print(' theta taken from 2M2224 case with Mie clouds')
     print(' crystalline enstatite slab + rust deck')
     print('------------------------------------------------------')
     print('mean value of modelspectrum / L benchmark = '+str(np.mean(difference_spectrum)))
     print('std deviation of modelspectrum / L benchmark = '+str(np.std(difference_spectrum)))
     percent_change = np.mean(abs(outspec - obspec[1,:])/obspec[1,:])

     if (percent_change < 0.01):
          print("less than 1percent difference with L dwarf regime benchmark")
          print('-------------------------------------------------------')
          return True
     else:
          print("greater than 1 percent difference with L dwarf regime benchmark")
          print('-------------------------------------------------------')
          return False

