#!/bin/tcsh
#SBATCH --job-name=W0751_PLSC_Prof7        # Job name
#SBATCH --mail-type=ALL                        # Mail events: BEGIN, END, FAIL
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=32                    # Tasks per node
#SBATCH --time=150:00:00                        # Walltime (hh:mm:ss)
#SBATCH --output=W0751_GreySlab_Prof7.out       # Combined stdout/stderr
#SBATCH --partition=core32                      # Queue/partition
#SBATCH --mem=190000M

# --- Record start time in seconds since epoch ---
set start_epoch=`date +%s`
set time_start=`date '+%T %d_%h_%y'`

echo "------------------------------------------------------"
echo "Job is running on nodes: $SLURM_JOB_NODELIST"
echo "------------------------------------------------------"
echo "SLURM: submitting host is $SLURM_SUBMIT_HOST"
echo "SLURM: working directory is $SLURM_SUBMIT_DIR"
echo "SLURM: job ID is $SLURM_JOB_ID"
echo "SLURM: job name is $SLURM_JOB_NAME"
echo "SLURM: partition is $SLURM_JOB_PARTITION"
echo "SLURM: number of nodes is $SLURM_JOB_NUM_NODES"
echo "SLURM: tasks per node is $SLURM_NTASKS_PER_NODE"
echo "------------------------------------------------------"

# --- Conda environment ---
# source /beegfs/car/jess/miniconda3/etc/profile.d/conda.sh
# conda activate brewster311
source ~/.tcshrc
eval `/usr/bin/modulecmd tcsh load use.own`


# --- Working directory ---
setenv WDIR /home2/jess/Brewster/brewster-YDwarfs
setenv PATH ${PATH}:${WDIR}
setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${WDIR}
setenv PYTHONPATH ${WDIR}

cd ${WDIR}

# --- Load OpenMPI module ---
eval `/usr/bin/modulecmd tcsh load openmpi-4.0.5`



# --- MPI run ---
mpirun --map-by ppr:32:node -x PATH -x PYTHONPATH -x LD_LIBRARY_PATH \
    python W0751_PLSC_Prof7.py \
    > /beegfs/car/jess/W0751/TEST_PLSC_Prof7.log

# --- Compute elapsed time ---
set end_epoch=`date +%s`
set time_end=`date '+%T %d_%h_%y'`
@ elapsed_sec = $end_epoch - $start_epoch

@ days    = $elapsed_sec / 86400
@ hours   = ( $elapsed_sec % 86400 ) / 3600
@ minutes = ( $elapsed_sec % 3600 ) / 60

echo "Started at: $time_start"
echo "Ended at:   $time_end"
echo "Elapsed:    ${days}d:${hours}h:${minutes}m"
echo "------------------------------------------------------"
echo "Job ends"
