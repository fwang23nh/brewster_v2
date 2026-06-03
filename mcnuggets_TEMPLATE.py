#!/usr/bin/env python

""" McNuggets: the post-processing tool for brewster"""
from __future__ import print_function

from builtins import str
from builtins import range
import numpy as np
import scipy as sp
import test_module
import utils
import ciamod
import TPmod
import nugbits_TEMPLATE as nb
import settings
import os
import gc
import sys
import pickle
import time
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from mpi4py import MPI

__author__ = "Fei Wang, Ben Burningham"
__copyright__ = "Copyright 2026 - Ben Burningham"
__credits__ = ["Ben Burningham","The EMCEE DOCS"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Ben Burningham"
__email__ = "burninghamster@gmail.com"
__status__ = "Development"

def split(container, count):
#    """
#    Simple function splitting a container into equal length chunks.
#    Order is not preserved but this is potentially an advantage depending on
#    the use case.
#    """
    return [container[_i::count] for _i in range(count)]


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# set up parallel bits
    # Start by getting the chain, then set up the run arguments, then loop for Teff etc
settings.init()

world_comm = MPI.COMM_WORLD
node_comm = world_comm.Split_type(MPI.COMM_TYPE_SHARED)
rank = node_comm.Get_rank()


COMM = MPI.COMM_WORLD


# # only the first rank has to do all the crap...
# runname "YOUR RUNNAME HERE"

# only the first rank has to do all the crap...
# runname = sys.argv[1] #"WISE0146_P7_NC"

runname = "vosimp0136_maxtheta"

# # using the Burrow's alkalis (via Mike)?
# malk = 0

#Are we testing?
testrun = 0

# length of test??
testlen = 1000

# OK finish?
fin = 0

# # what's the error on the distance?
# sigDist = 0.1

# # What's the error on the photometry used to flux calibrate the data?
# sigPhot = 0.02

# what's the error on the distance?
sigDist = 0.02 #sys.argv[2]

# What's the error on the photometry used to flux calibrate the data?
sigPhot = 0.02 #sys.argv[3] #0.02

# Where are the pickles stored?
outdir = "/beegfs/car/fei/vosimp0136/"

# which opacity set did we use?
# xlist = "gaslistR10K.dat"

# Where are the cross sections?
# give the full path
# xpath = "/beegfs/car/bb/Linelists/"


# If we're corrcting the distance (e.g for Gaia update),
# put it here, or or comment out 
# dist = 11.56

# But we want to a much wider wavelength range...
w1 = 0.7
w2 = 20.0

argfile =outdir+runname+"_runargs.pic"
args_instance = utils.pickle_load(argfile)
args_instance.w1=w1
args_instance.w2=w2
inlinetemps,inwavenum,gasnames,gasmass,nwave=utils.get_gasdetails(args_instance.gaslist, args_instance.w1,args_instance.w2,args_instance.xpath,args_instance.xlist)
args_instance.inlinetemps=inlinetemps
args_instance.inwavenum=inwavenum
args_instance.nwave=nwave

tmpcia, ciatemps = ciamod.read_cia("data/CIA_DS_aug_2015.dat",args_instance.inwavenum)
ciatemps = np.asfortranarray(ciatemps, dtype='float32')
args_instance.ciatemps=ciatemps
settings.init(args_instance)

settings.cia, _ = utils.shared_memory_array(rank, node_comm, (4,ciatemps.size,nwave), datatype='f')
if (rank == 0):
    # cia = np.asfortranarray(np.empty((4,ciatemps.size,nwave)),dtype='float32')
    settings.cia[:,:,:] =  tmpcia[:,:,:nwave].copy() 


# set up shared memory array for linelist
ngas = len(args_instance.gaslist)
npress= args_instance.press.size
ntemps = args_instance.inlinetemps.size
settings.linelist, _ = utils.shared_memory_array(rank, node_comm, (ngas,npress,ntemps,args_instance.nwave), datatype='f')

if (rank == 0):
# Now we'll get the opacity files into an array
    settings.linelist[:,:,:,:] = utils.get_opacities(args_instance.gaslist,args_instance.w1,args_instance.w2,args_instance.press,args_instance.xpath,args_instance.xlist,args_instance.malk)


with open(outdir+runname+'_configs.pic', 'rb') as file:
    configs= pickle.load(file)
re_params=configs['re_params']

if re_params.dictionary['cloud']:
    cloudfile =outdir+runname+"_cloudata.pic"
    cloudata = utils.pickle_load(cloudfile)

    ncloud = len(args_instance.cloudname_set)
    nmiewave= args_instance.miewave.size
    nmierad= args_instance.mierad.size
    settings.cloudata, _ = utils.shared_memory_array(rank, node_comm, (ncloud,3,nmiewave,nmierad))

    if (rank == 0):
    # Now we'll get the opacity files into an array
        settings.cloudata[:,:,:,:] = cloudata

else:#send empty clouddata (clear atmosphere) to settings for model_spec calculation
    settings.cloudata=np.zeros((0,0), dtype=float)

world_comm.Barrier()


# that's all the input.. .off we go...
rawsamples, probs,ndim = nb.get_endchain(outdir+runname,fin)
slen = rawsamples.shape[0]

if (testrun == 1):
    settings.samples = rawsamples[slen-testlen:,:]
    if (world_comm.rank == 0):
        print('testing with '+str(testlen)+' samples')
        print('full run would be '+str(slen))
else:
    settings.samples = rawsamples
    
slen = settings.samples.shape[0]
print("new settings generated,start sampling")


# Collect whatever has to be done in a list. Here we'll just collect a list of
# numbers. Only the first rank has to do this.
if world_comm.rank == 0:
    jobs = list(range(slen))
    t1 = time.process_time()

    # Split into however many cores are available.
    jobs = split(jobs, COMM.size)
else:
    jobs = None

# Scatter jobs across cores.
jobs = world_comm.scatter(jobs, root=0)

# Now each rank just does its jobs and collects everything in a results list.
# Make sure to not use super big objects in there as they will be pickled to be
# exchanged over MPI.
results = []

for job in jobs:
    res = nb.teffRM(settings.samples[job,0:ndim],re_params,sigDist,sigPhot)
    gc.collect()
    results.append(res)

# Gather results on rank 0.
results = MPI.COMM_WORLD.gather(results, root=0)

if world_comm.rank == 0:
    # Flatten list of lists.
    results = [_i for tmp in results for _i in tmp]

    print("writing results to samplus")
    runtime = time.process_time() - t1 
    print('time taken = '+str(runtime))
    samplus = np.array(results)

    save_object(samplus,outdir+runname+'_postprod.pic')

