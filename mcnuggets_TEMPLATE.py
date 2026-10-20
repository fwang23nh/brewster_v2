#!/usr/bin/env python

"""McNuggets: MPI post-processing of Brewster MCMC retrievals."""
import os
from pathlib import Path
import pickle
import time
import traceback
import numpy as np
from mpi4py import MPI
import utils
import ciamod
import nugbits_TEMPLATE as nb
import settings

__author__ = "Fei Wang, Ben Burningham"
__copyright__ = "Copyright 2026 - Ben Burningham"
__credits__ = ["Ben Burningham","The EMCEE DOCS"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Ben Burningham"
__email__ = "burninghamster@gmail.com"
__status__ = "Development"

# Edit these values for the retrieval and machine being used.
runname = 'vosimp0136_maxtheta'
outdir = '/beegfs/car/fei/vosimp0136/'
fin = 0                         # 0: snapshot; 1: finished sampler
sigDist = 0.02                   # Distance uncertainty / pc
sigPhot = 0.02                   # Absolute flux calibration uncertainty / mag
w1,w2 = 0.7,20.0                # Finite wavelength interval / micron
testrun = 0
testlen = 1000
xpath = None                    # Override the saved opacity directory if needed.
xlist = None                    # Override the saved gas catalogue if needed.


def split(container,count):
    return [container[i::count] for i in range(count)]


def save_object(obj,filename):
    with open(filename,'wb') as output:
        pickle.dump(obj,output,pickle.HIGHEST_PROTOCOL)


def main():
    world_comm = MPI.COMM_WORLD
    node_comm = world_comm.Split_type(MPI.COMM_TYPE_SHARED)
    rank = node_comm.Get_rank()
    source_dir = Path(__file__).resolve().parent
    prefix = str(Path(outdir) / runname)
    args_instance = utils.pickle_load(prefix + '_runargs.pic')
    configs = utils.pickle_load(prefix + '_configs.pic')
    re_params = configs['re_params']
    if re_params.samplemode != 'mcmc':
        raise ValueError('This driver loads MCMC chains, not MultiNest samples.')
    if testrun not in (0,1) or (testrun == 1 and testlen <= 0):
        raise ValueError('testrun must be 0 or 1; testlen must be positive.')
    if not np.all(np.isfinite([w1,w2])) or not 0 < w1 < w2:
        raise ValueError('Require 0 < w1 < w2 in micron.')
    args_instance.w1,args_instance.w2 = w1,w2
    # Resolve relative saved paths against Brewster, independent of launch CWD.
    opacity_dir = Path(xpath if xpath is not None else args_instance.xpath)
    if not opacity_dir.is_absolute():
        opacity_dir = source_dir / opacity_dir
    if not opacity_dir.is_dir():
        raise FileNotFoundError('Set xpath to the opacity directory: ' + str(opacity_dir))
    args_instance.xpath = str(opacity_dir.resolve()) + os.sep
    catalogue = Path(xlist if xlist is not None else args_instance.xlist)
    if not catalogue.is_absolute():
        catalogue = source_dir / catalogue
    args_instance.xlist = str(catalogue)

    # Read the large opacity file for grid details only once per node.
    details = None
    if rank == 0:
        details = utils.get_gasdetails(args_instance.gaslist,w1,w2,
                                      args_instance.xpath,args_instance.xlist)
    (args_instance.inlinetemps,args_instance.inwavenum,args_instance.gasnames,
     args_instance.gasmass,nwave) = node_comm.bcast(details,root=0)
    args_instance.nwave = nwave
    args_instance.inwavenum = np.asfortranarray(args_instance.inwavenum,dtype='float64')

    cia_file = source_dir / 'data' / 'CIA_DS_aug_2015.dat'
    with cia_file.open() as file:
        for _ in range(3):
            next(file)
        args_instance.nciawave,args_instance.nciatemps = map(int,next(file).split())
    args_instance.ciatemps = np.zeros(args_instance.nciatemps,dtype='float32',order='F')
    settings.init(args_instance)
    settings.cia,cia_win = utils.shared_memory_array(
        rank,node_comm,(4,args_instance.nciatemps,nwave),datatype='f')
    cia_win.Lock_all()
    if rank == 0:
        ciamod.read_cia(str(cia_file),args_instance.nciawave,
                        args_instance.inwavenum,settings.cia,args_instance.ciatemps)
    args_instance.ciatemps = node_comm.bcast(args_instance.ciatemps,root=0)

    ngas = len(args_instance.gaslist)
    npress = args_instance.press.size
    ntemps = args_instance.inlinetemps.size
    settings.linelist,line_win = utils.shared_memory_array(
        rank,node_comm,(ngas,npress,ntemps,nwave),datatype='f')
    line_win.Lock_all()
    if rank == 0:
        settings.linelist[:,:,:,:] = utils.get_opacities(
            args_instance.gaslist,w1,w2,args_instance.press,
            args_instance.xpath,args_instance.xlist,args_instance.malk)

    windows = [cia_win,line_win]  # Keep every shared allocation alive through sampling.
    if re_params.dictionary['cloud']:
        cloud_shape = (len(args_instance.cloudname_set),3,
                       args_instance.miewave.size,args_instance.mierad.size)
        settings.cloudata,cloud_win = utils.shared_memory_array(rank,node_comm,cloud_shape)
        cloud_win.Lock_all()
        windows.append(cloud_win)
        if rank == 0:
            settings.cloudata[:,:,:,:] = utils.pickle_load(prefix + '_cloudata.pic')
    else:
        settings.cloudata = np.zeros((0,0),dtype=float)
    for win in windows:
        win.Sync()
    node_comm.Barrier()
    for win in windows:
        win.Sync()

    # Only the world root reads the chain; each worker receives its own rows.
    jobs = None
    if world_comm.rank == 0:
        rawsamples,probs,ndim = nb.get_endchain(prefix,fin)
        if testrun == 1:
            rawsamples = rawsamples[-testlen:]
        print('Post-processing',len(rawsamples),'samples over',w1,'to',w2,'micron',flush=True)
        jobs = split(list(enumerate(rawsamples)),world_comm.size)
        t1 = time.perf_counter()
    jobs = world_comm.scatter(jobs,root=0)
    results = []
    for job,theta in jobs:
        result = nb.teffRM(theta,re_params,sigDist,sigPhot)
        results.append((job,result))
    results = world_comm.gather(results,root=0)
    if world_comm.rank == 0:
        results = [row for chunk in results for row in chunk]
        results.sort(key=lambda row: row[0])
        samplus = np.array([row[1] for row in results])
        save_object(samplus,prefix + '_postprod.pic')
        print('Saved',prefix + '_postprod.pic',
              'in',time.perf_counter()-t1,'seconds',flush=True)
    world_comm.Barrier()
    # Shared arrays are no longer used after this point.
    for win in windows:
        win.Unlock_all()
        win.Free()
    node_comm.Free()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # A failed worker must stop its peers, not leave them hanging in a collective.
        traceback.print_exc()
        if MPI.COMM_WORLD.size > 1:
            MPI.COMM_WORLD.Abort(1)
        raise
