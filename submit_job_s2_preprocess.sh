#!/bin/bash -l
#PBS -N get_data
#PBS -A UWAS0120
#PBS -l select=1:ncpus=16:mem=64GB
#PBS -l walltime=02:30:00
#PBS -q casper
#PBS -j oe
#PBS -o /glade/derecho/scratch/zxhua/dl_hazard_guidance/job_log

### Load your conda library
module load conda
conda activate dl_xesmf_2

### Run job
cd /glade/derecho/scratch/zxhua/dl_hazard_guidance/process_fcst/
python s2_24hr_report_dataset_multiprocessing.py ./process_fcst_config/job_s2_Fengwu_preprocess.yaml
# python s2_24hr_report_local_condition_dataset.py ./process_fcst_config/job_s2_Fengwu_preprocess_zarr.yaml