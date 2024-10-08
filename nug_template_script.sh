#PBS -S /bin/tcsh
#PBS -N YOURJOBNAME
#PBS -m abe
#PBS -l nodes=4:ppn=32
#PBS -l walltime=20:00:00
#PBS -k oe
#PBS -q main

source ~/.tcshrc
module load use.own
module unload mpich2-x86_64
module load mpich2-intel

setenv WDIR /path/to/your/WDIR

setenv PATH ${PATH}:${WDIR}
setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${WDIR}

py3up

unlimit stacksize

limit coredumpsize 0

set time_start=`date '+%T%t%d_%h_06'`
  
echo ------------------------------------------------------
echo -n 'Job is running on node '; cat $PBS_NODEFILE
echo ------------------------------------------------------
echo PBS: qsub is running on $PBS_O_HOST
echo PBS: originating queue is $PBS_O_QUEUE
echo PBS: executing queue is $PBS_QUEUE
echo PBS: working directory is $PBS_O_WORKDIR
echo PBS: execution mode is $PBS_ENVIRONMENT
echo PBS: job identifier is $PBS_JOBID
echo PBS: job name is $PBS_JOBNAME
echo PBS: node file is $PBS_NODEFILE
echo PBS: current home directory is $PBS_O_HOME
echo PBS: PATH = $PBS_O_PATH
echo ------------------------------------------------------



cd ${WDIR}

mpirun -env I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=1 \
       -machinefile $PBS_NODEFILE -n 128 -ppn 32 \
       python YOUR_mcnuggets_name.py > path/to/your/logfile.txt



set time_end=`date '+%T%t%d_%h_06'`
echo Started at: $time_start
echo Ended at: $time_end
echo ------------------------------------------------------
echo Job ends

#mpdallexit
