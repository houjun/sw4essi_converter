#!/bin/bash
#SBATCH -A m3354
#SBATCH -N 16
#SBATCH -p debug
#SBATCH -t 00:30:00 
# #SBATCH -p regular
# #SBATCH -t 01:20:00 
#SBATCH -C knl,quad,cache
#SBATCH -L SCRATCH 
#SBATCH -o o%j.ssi2hdf5
#SBATCH -e o%j.ssi2hdf5
#SBATCH -J ssi2hdf5

# #DW jobdw capacity=4000GB access_mode=striped type=scratch
# #DW stage_in source=/global/cscratch1/sd/houhun/stripe_large/surface-800-100.ssi destination=$DW_JOB_STRIPED/surface-800-100.ssi type=file
# #DW stage_out source=$DW_JOB_STRIPED/rechdf5_output.h5 destination=/global/cscratch1/sd/houhun/rechdf5_output.h5 type=file

INFILE=/global/cscratch1/sd/houhun/stripe_large/surface-800-100.ssi
OUTFILE=/global/cfs/cdirs/m3354/tang/eqsim-data/data/Data_1km_2021_SEPT_M7_10Hz_Vsmin_140/rechdf5_output.h5
# /global/cscratch1/sd/houhun/stripe_large/rechdf5_output.h5
OUTDIR=/global/cfs/cdirs/m3354/tang/eqsim-data/data/Data_1km_2021_SEPT_M7_10Hz_Vsmin_140

date

source activate myenv

stdbuf -o0 -e0 -i0 srun -N 16 -n 119 python /global/u1/h/houhun/sw4essi_converter/ssi2sachdf5.py -i $INFILE -o $OUTFILE -d $OUTDIR -x 1000 120000 -y 1000 80000 -s 1000 -v

date
