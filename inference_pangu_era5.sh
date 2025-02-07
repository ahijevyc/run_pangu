#!/bin/bash -l
#PBS -N infer_pangu
#PBS -A NMMM0021
#PBS -l select=1:ncpus=4:mpiprocs=1:ngpus=1:mem=32GB
#PBS -l gpu_type=a100
#PBS -l walltime=01:00:00
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

python inference_to_nc_iterative_era5.py --start_date 20240501 --end_date 20240501 --inference_input_dir pangu_era5_input_data --inference_output_dir pangu_era5_forecast_data
