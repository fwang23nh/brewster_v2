#!/usr/bin/env python
"""MCMC Retrieval Setup Template"""
import os
import utils
import numpy as np
import retrieval_run
import settings


__author__ = "Fei Wang"
__copyright__ = "Copyright 2024 - Fei Wang"
__credits__ = ["Fei Wang", "Ben Burningham"]
__license__ = "GPL"
__version__ = "0.2"  
__maintainer__ = ""
__email__ = ""
__status__ = "Development"


fwhm=555.0
wavelength_range=[1.0,12]
ndata=1
wavpoints=None
R_file = 'GJ499C_R_file.txt'

chemeq=0
gaslist =  ['h2o','co','co2','ch4','sio','tio','vo','crh','feh','k','na']
gastype_list=['U','U','U','U','U','U','U','N','N','U','U']
ptype=7
do_clouds=1
npatches=1
cloudname = ['Mie scattering cloud slab--SiO','Mie scattering cloud slab--MgSiO3', 'Mie scattering cloud deck--Fe']  
cloudpacth_index=[[1],[1],[1]] 
particle_dis=['hansan','hansan','hansan']


# cloudname = []  
# cloudpacth_index=[] 


# particle_dis=['hansan','log_normal']
# cloudname = ['power law cloud slab']  
do_bff=1

do_fudge=1
samplemode='mcmc'
# samplemode='multinest'

instrument_instance = utils.Instrument(fwhm,wavelength_range,ndata,wavpoints, R_file)
re_params = utils.Retrieval_params(samplemode,chemeq,gaslist,gastype_list,fwhm,do_fudge,ptype,do_clouds,npatches,cloudname,cloudpacth_index,particle_dis,instrument_instance)
model_config_instance = utils.ModelConfig(samplemode,do_fudge)
io_config_instance = utils.IOConfig()


io_config_instance.outdir="/beegfs/general/viktoria/brewster_v2/GJ499C_results/mcmc_GJ_slabsio_slabmgsio3_deckfe_ptype7_no_inversion_nonuni_crh/"
io_config_instance.runname='mcmc_GJ_slabsio_slabmgsio3_deckfe_ptype7_no_inversion_nonuni_crh'
io_config_instance.update_dictionary()


model_config_instance.dist= 1000/50.23
model_config_instance.xlist ='gaslistR10K.dat'
model_config_instance.xpath = "/beegfs/general/viktoria/Linelists/"
model_config_instance.do_bff=1
model_config_instance.malk=0 #0 allard 1 burrows
#model_config_instance.ch4=0
model_config_instance.niter=50000
model_config_instance.update_dictionary()



re_params.dictionary['gas']['h2o']['params']['log_abund']['distribution']=['normal',-3.8,0.5]
re_params.dictionary['gas']['co']['params']['log_abund']['distribution']=['normal', -3.5,0.5]
re_params.dictionary['gas']['co2']['params']['log_abund']['distribution']=['normal',-7.45,0.5]
re_params.dictionary['gas']['ch4']['params']['log_abund']['distribution']=['normal',-5.9,0.5]
re_params.dictionary['gas']['sio']['params']['log_abund']['distribution']=['normal',-4.5,0.5]
re_params.dictionary['gas']['tio']['params']['log_abund']['distribution']=['normal',-8.2,0.5]
re_params.dictionary['gas']['vo']['params']['log_abund']['distribution']=['normal',-9.1,0.5]
re_params.dictionary['gas']['crh']['params']['log_abund']['distribution']=['normal',-7.9,0.5]
re_params.dictionary['gas']['feh']['params']['log_abund']['distribution']=['normal',-6.3,0.5]
#re_params.dictionary['gas']['k']['params']['log_abund']['distribution']=['normal',-5.5,0.5]
#re_params.dictionary['gas']['na']['params']['log_abund']['distribution']=['normal',-5.5,0.5]
re_params.dictionary['gas']['K_Na']['params']['log_abund']['distribution']=['normal',-5.5,0.5]
re_params.dictionary['refinement_params']['params']['logg']['distribution']=['normal',5.0,0.5]
re_params.dictionary['cloud']['patch 1']['Mie scattering cloud slab--MgSiO3']['params']['logp_mcs_MgSiO3']['distribution']=['normal',-3.5,0.2]
re_params.dictionary['cloud']['patch 1']['Mie scattering cloud slab--MgSiO3']['params']['dp_mcs_MgSiO3']['distribution']=['normal',0.2,0.2]
re_params.dictionary['cloud']['patch 1']['Mie scattering cloud slab--MgSiO3']['params']['hansan_a_mcs_MgSiO3']['distribution']=['normal',1.5,0.2]
re_params.dictionary['cloud']['patch 1']['Mie scattering cloud slab--MgSiO3']['params']['hansan_b_mcs_MgSiO3']['distribution']=['normal',0.2,0.2]
#re_params.dictionary['cloud']['patch 1']['Mie scattering cloud slab--Mg2SiO4']['params']['logp_mcs_Mg2SiO4']['distribution']=['normal',-1.5,0.5]
re_params.dictionary['pt']['params']['Tint']['distribution']=['normal', 2200,200]
re_params.dictionary['pt']['params']['alpha']['distribution']=['normal', 1.60,0.1]
re_params.dictionary['pt']['params']['lndelta']['distribution']=['normal', -3.0,0.5]
re_params.dictionary['pt']['params']['T1']['distribution']=['normal', 200,200]
re_params.dictionary['pt']['params']['T2']['distribution']=['normal', 400,200]
re_params.dictionary['pt']['params']['T3']['distribution']=['normal', 1200,200]

re_params.dictionary['cloud']['patch 1']['Mie scattering cloud slab--SiO']['params']['logp_mcs_SiO']['distribution']=['normal',-1.5,0.5]
re_params.dictionary['cloud']['patch 1']['Mie scattering cloud slab--SiO']['params']['dp_mcs_SiO']['distribution']=['normal',0.3,0.5]
re_params.dictionary['cloud']['patch 1']['Mie scattering cloud slab--SiO']['params']['hansan_a_mcs_SiO']['distribution']=['normal',-1.8,0.5]
re_params.dictionary['cloud']['patch 1']['Mie scattering cloud slab--SiO']['params']['hansan_b_mcs_SiO']['distribution']=['normal',0.2,0.5]

re_params.dictionary['cloud']['patch 1']['Mie scattering cloud deck--Fe']['params']['logp_mcd_Fe']['distribution']=['normal',1.0,0.2]
re_params.dictionary['cloud']['patch 1']['Mie scattering cloud deck--Fe']['params']['dp_mcd_Fe']['distribution']=['normal',0.2,0.2]
re_params.dictionary['cloud']['patch 1']['Mie scattering cloud deck--Fe']['params']['hansan_a_mcd_Fe']['distribution']=['normal',-2.4,0.2]
re_params.dictionary['cloud']['patch 1']['Mie scattering cloud deck--Fe']['params']['hansan_b_mcd_Fe']['distribution']=['normal',0.4,0.2]
#re_params.dictionary['pt']['params']['alpha']['distribution']=['normal', 2.0,0.1]
#re_params.dictionary['pt']['params']['lndelta']['distribution']=['normal', 2.0,0.1]
#re_params.added_params('tolerance_parameter_2')
#re_params.dictionary['added_params']['tolerance_parameter_2']['distribution']=['normal',0.1,0.5]
#re_params.update_dictionary()


obspec = np.asfortranarray(np.loadtxt("GJ499C_spectrum.txt",dtype='d',unpack='true')) # G570D_2MassJcalib.dat
args_instance = utils.ArgsGen(re_params,model_config_instance,instrument_instance,obspec)
settings.init(args_instance)
retrieval_run.brewster_reterieval_run(re_params,model_config_instance,io_config_instance)

