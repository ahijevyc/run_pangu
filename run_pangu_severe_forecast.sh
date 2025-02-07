#!/bin/bash -l

#PBS -N run_pangu_severe
#PBS -A NMMM0021
#PBS -l select=1:ncpus=7:mpiprocs=1:ngpus=1:mem=25GB
#PBS -l gpu_type=a100
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

#yyyymmddhh=$1
run_dir=/glade/derecho/scratch/sobash/pangu_realtime/$yyyymmddhh/$ic
mkdir -pv $run_dir

# download ecmwf data (not needed for ERA5 ICs from RDA)
python s1_get_ecmwf.py --start_date $yyyymmddhh --end_date $yyyymmddhh --output_dir $run_dir --ic $ic

# process ecmwf data into format required for Pangu
python s2_make_ic_ecmwf.py --start_date $yyyymmddhh --end_date $yyyymmddhh --output_dir $run_dir --ic $ic

# run Pangu with ecmwf ICs
python s3_run_pangu_ecmwf.py --start_date $yyyymmddhh --end_date $yyyymmddhh --inference_input_dir $run_dir --inference_output_dir $run_dir/pangu_forecast_data --ic $ic

# upscale pangu output
# this uses 6 processors
python s4_get_pangu_infer_init_upscaled_fcst.py ./fcst_config/Download_Pangu_hresan_init_upscaled_fcst.yaml --start_date $yyyymmddhh --end_date $yyyymmddhh --ic $ic

# convert upscaled data into parquet files
# this uses 1 processor
python s5_24hr_report_dataset_multiprocessing.py ./fcst_config/job_s2_PanguHRESAN_preprocess.yaml --start_date $yyyymmddhh --end_date $yyyymmddhh --ic $ic

cd /glade/work/sobash/run_pangu/pytorch_project

# run inference with trained transformer model
#./trained_from_2018/20250117_225144/ FengWu model
python evaluate_seq_days_feature_concat_realtime.py ./logs/trained_from_2018/20241211_122629 --start_date $yyyymmddhh --end_date $yyyymmddhh --ic $ic --model pangu
