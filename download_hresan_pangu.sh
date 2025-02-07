#!/bin/bash -l
#PBS -N get_data
#PBS -A NAML0001
#PBS -l select=1:ncpus=16:mem=64GB
#PBS -l walltime=06:20:00
#PBS -q casper
#PBS -j oe
#PBS -o /glade/derecho/scratch/zxhua/AI_global_forecast_model_for_education/Pangu_Weather_for_copy/job_log

### Load your conda library
module load cuda/11.8.0
module load cudnn/8.7.0.84
module load conda
conda activate dl_xesmf_2

### Run job
cd /glade/derecho/scratch/zxhua/AI_global_forecast_model_for_education/Pangu_Weather_for_copy
python download_hresan_rda_timedelta.py --start_date 20240101 --end_date 20241001 --output_dir pangu_hresan_input_data_18utc --time_delta -6

