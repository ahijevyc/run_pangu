#!/bin/csh

cat <<END | qsub
#!/bin/csh
#PBS -N ${1}fengwu
#PBS -A NMMM0021
#PBS -l select=1:ncpus=1:mpiprocs=1:ngpus=1:mem=10GB
#PBS -l gpu_type=v100
#PBS -l walltime=01:45:00
#PBS -q casper

### Load your conda library
module restore fengwu
conda activate pangu

### Run job
cd /glade/work/ahijevyc/run_pangu

python fengwu.py $1
END

