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
python get_ecmwf.py 2025012000

python download_euro.py --start_date 2025012000 --end_date 2025012000 --output_dir pangu_hres_input_data
