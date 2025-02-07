#!/bin/bash -l
#PBS -N get_data
#PBS -A NAML0001
#PBS -l select=1:ncpus=1:mem=8GB
#PBS -l walltime=08:30:00
#PBS -q casper
#PBS -j oe
#PBS -o /glade/derecho/scratch/zxhua/AI_global_forecast_model_for_education/Pangu_Weather_for_copy/job_log

### Load your conda library
module load cuda/11.8.0
module load cudnn/8.7.0.84
module load conda
conda activate pangu_test

### Run job
cd /glade/derecho/scratch/zxhua/AI_global_forecast_model_for_education/Pangu_Weather_for_copy
python download_era5_rda_timedelta.py --start_date 20230101 --end_date 20231231 --output_dir pangu_era5_input_data_18utc --time_delta -6

