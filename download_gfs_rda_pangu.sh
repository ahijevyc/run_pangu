#!/bin/bash -l
#PBS -N prep_gfs
#PBS -A NMMM0021
#PBS -l select=1:ncpus=1:mem=12GB
#PBS -l walltime=10:00:00
#PBS -q casper
#PBS -j oe
#PBS -o /glade/work/sobash/run_pangu/job_log

### Load your conda library
module load cuda/11.8.0
module load cudnn/8.7.0.84
module load conda
conda activate ainwp

### Run job
cd /glade/work/sobash/run_pangu/
python download_gfs_rda.py --start_date 20250118 --end_date 20250118 --output_dir pangu_gfs_input_data

