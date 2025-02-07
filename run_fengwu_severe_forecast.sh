#!/bin/bash -l

#PBS -N run_pangu_severe
#PBS -A NMMM0021
#PBS -l select=1:ncpus=7:mpiprocs=1:ngpus=1:mem=25GB
#PBS -l gpu_type=a100
#PBS -l walltime=00:10:00
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

# make ICs 6 hours before this date for fengwu
yyyymmddhh2=$(date -d "${yyyymmddhh:0:8} ${yyyymmddhh:8:2} - 6 hours" +%Y%m%d%H)

#yyyymmddhh=$1
run_dir=/glade/derecho/scratch/sobash/fengwu_realtime/$yyyymmddhh/$ic
mkdir -pv $run_dir

# download ecmwf data (not needed for ERA5 ICs from RDA)
#python s1_get_ecmwf.py --start_date $yyyymmddhh --end_date $yyyymmddhh --output_dir $run_dir --ic $ic 
python s1_get_ecmwf.py --start_date $yyyymmddhh2 --end_date $yyyymmddhh2 --output_dir $run_dir --ic $ic

# process ecmwf data into format required for Pangu
#python s2_make_ic_ecmwf.py --start_date $yyyymmddhh --end_date $yyyymmddhh --output_dir $run_dir --ic $ic
python s2_make_ic_ecmwf.py --start_date $yyyymmddhh2 --end_date $yyyymmddhh2 --output_dir $run_dir --ic $ic

# first file should already be there if pangu forecasts ran beforehand
ln -sf /glade/derecho/scratch/sobash/pangu_realtime/$yyyymmddhh/$ic/pangu_$ic_init_$yyyymmddhh.nc ${run_dir}/.

# run Pangu with ecmwf ICs
#python s3_run_pangu_ecmwf.py --start_date $yyyymmddhh --end_date $yyyymmddhh --inference_input_dir $run_dir --inference_output_dir $run_dir/pangu_forecast_data --ic $ic
python s3_run_fengwu_ecmwf.py --start_date $yyyymmddhh --end_date $yyyymmddhh \
    --inference_input_dir $run_dir --inference_input_dir_minus_6 $run_dir --inference_output_dir $run_dir/fengwu_forecast_data \
    --model_dir /glade/derecho/scratch/zxhua/AI_global_forecast_model_for_education/FengWu/model \
    --ic $ic

# upscale pangu output
# this uses 6 processors
python s4_get_fengwu_infer_init_upscaled_fcst.py ./fcst_config/Download_Fengwu_hresan_init_upscaled_fcst.yaml --start_date $yyyymmddhh --end_date $yyyymmddhh --ic $ic

# convert upscaled data into parquet files
# this uses 1 processor
python s5_24hr_report_dataset_multiprocessing_fengwu.py ./fcst_config/job_s2_FengWu_preprocess.yaml --start_date $yyyymmddhh --end_date $yyyymmddhh --ic $ic

cd /glade/work/sobash/run_pangu/pytorch_project

# run inference with trained transformer model
python evaluate_seq_days_feature_concat_realtime.py ./logs/trained_from_2018/20250117_225144 --start_date $yyyymmddhh --end_date $yyyymmddhh --ic $ic --model fengwu
#python evaluate_seq_days_feature_concat_realtime.py ./logs/trained_from_2018/20241211_122629 --start_date $yyyymmddhh --end_date $yyyymmddhh --ic $ic --model pangu
