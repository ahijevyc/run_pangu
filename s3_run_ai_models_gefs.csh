#!/bin/csh

set yyyymmddhh=$1
if ($yyyymmddhh < 100000000) then
    echo expected yyyymmddhh
    exit 1
endif
cat <<EOF|qsub
#!/bin/bash -l
#PBS -N $yyyymmddhh
#PBS -A NMMM0021
#PBS -l select=1:ncpus=1:mpiprocs=1:ngpus=1:mem=25GB
#PBS -l gpu_type=a100
#PBS -l walltime=01:00:00
#PBS -q casper@casper-pbs

### Load your conda library
module purge
module load ncarenv/24.12
module reset
module load cudnn/9.2.0.82-12 conda
conda activate pangu
module list

### Run job
cd $WORK/run_pangu/

./ai_models_gefs.csh $yyyymmddhh panguweather
EOF
