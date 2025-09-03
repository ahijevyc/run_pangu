#!/bin/bash -l
#PBS -N grphcst
#PBS -A NMMM0021
#PBS -l select=1:ncpus=1:mpiprocs=1:ngpus=1:mem=33GB
#PBS -l gpu_type=a100
#PBS -l walltime=00:36:00
#PBS -q casper

### Load your conda library
module purge
source $SCRATCH/new_e2s_project/.venv/bin/activate

# Usage: 
# qsub -v DATE_ARG=20240530 s3_run_graphcast_gefs.sh
### Run job
uv run --active $WORK/run_pangu/s3_run_graphcast_gefs.py $DATE_ARG
