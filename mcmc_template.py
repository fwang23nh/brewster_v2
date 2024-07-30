#!/usr/bin/env python
"""MCMC Retrieval Setup Template"""

__author__ = "Fei Wang"
__copyright__ = "Copyright 2024 - Fei Wang"
__credits__ = ["Fei Wang", "Ben Burningham"]
__license__ = "GPL"
__version__ = "0.2"  
__maintainer__ = ""
__email__ = ""
__status__ = "Development"


import os 
import utils
import numpy as np
import retrieval_run
import settings


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
# do_bff=1

do_fudge=1
samplemode='mcmc'
# samplemode='multinest'

instrument_instance = utils.Instrument(fwhm,wavelength_range,ndata)
re_params = utils.Retrieval_params(samplemode,chemeq,gaslist,gastype_list,fwhm,do_fudge,ptype,do_clouds,npatches,cloudname,cloudpacth_index)
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

retrieval_run.brewster_reterieval_run(re_params,model_config_instance,io_config_instance)

