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
cd /glade/work/sobash/run_pangu

python s3_run_fengwu_ecmwf.py --start_date 2025020500 --end_date 2025020500 \
    --inference_input_dir /glade/derecho/scratch/sobash/fengwu_realtime/2025020500/hres/ \
    --inference_input_dir_minus_6 /glade/derecho/scratch/sobash/fengwu_realtime/2025020500/hres/ \
    --inference_output_dir /glade/derecho/scratch/sobash/fengwu_realtime/2025020500/hres/pangu_forecast_data \
    --model_dir /glade/derecho/scratch/zxhua/AI_global_forecast_model_for_education/FengWu/model
