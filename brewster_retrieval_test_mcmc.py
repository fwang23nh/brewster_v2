#!/usr/bin/env python

""" Module of processes to interpret cloud parameters from Brewster in testkit"""
# from __future__ import print_function
# import numpy as np
# import scipy as sp
# from scipy import interpolate
# from astropy.convolution import convolve, convolve_fft
# from astropy.convolution import Gaussian1DKernel

import os 
import utils
from collections import namedtuple
import numpy as np
import test_new_module
import brewster_run
import settings

__author__ = "Ben Burningham"
__copyright__ = "Copyright 2016 - Ben Burningham"
__credits__ = ["Ben Burningham","The EMCEE DOCS"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Ben Burningham"
__email__ = "burninghamster@gmail.com"
__status__ = "Development"


fwhm=700
wavelength_range=[1,2.8]
ndata=1
# wavpoints=None

chemeq=0
gaslist = ['h2o','co','tio','vo','crh','feh','na','k']
gastype_list=['U','U','U','U','U','N','U','U']

ptype=77  
do_clouds=1
npatches=2
cloudname = ['power law cloud slab','powerlaw cloud deck']  
cloudpacth_index=[[1],[1,2]] 


# cloudname = []  
# cloudpacth_index=[] 


# particle_dis=['hansan','log_normal']
# cloudname = ['power law cloud slab']  
do_fudge=1
# do_bff=1
samplemode='mcmc'

# samplemode='multinest'

instrument_instance = utils.Instrument(fwhm,wavelength_range,ndata)
re_params = utils.retrieval_params(samplemode,chemeq,gaslist,gastype_list,fwhm,do_fudge,ptype,do_clouds,npatches,cloudname,cloudpacth_index)
model_config_instance = utils.ModelConfig(samplemode)
io_config_instance = utils.IOConfig()


io_config_instance.outdir="/beegfs/car/fei/lsr1835/"
io_config_instance.runname='new_test'
io_config_instance.update_dictionary()


model_config_instance.dist= 5.689 
model_config_instance.update_dictionary()



obspec = np.asfortranarray(np.loadtxt("LSR1835_data_realcalib_new_trimmed.dat",dtype='d',unpack='true'))
settings.init()
settings.runargs=utils.args_gen(re_params,model_config_instance,instrument_instance,obspec)

brewster_run.brewster_reterieval_run(re_params,model_config_instance,io_config_instance)

