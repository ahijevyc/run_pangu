#!/bin/bash -l
#PBS -N infer_pangu
#PBS -A NMMM0021
#PBS -l select=1:ncpus=1:mpiprocs=1:ngpus=1:mem=32GB
#PBS -l gpu_type=v100
#PBS -l walltime=00:15:00
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

python s3_run_pangu_ecmwf.py --start_date 2024050300 --end_date 2024050300 --inference_input_dir ./ --inference_output_dir pangu_era5_forecast_data --ic era5
