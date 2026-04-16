#!/bin/bash
#SBATCH -A m3354
#SBATCH -N 16
# #SBATCH -p debug
# #SBATCH -t 00:30:00 
#SBATCH -p regular
#SBATCH -t 02:00:00 
#SBATCH -C knl,quad,cache
#SBATCH -L SCRATCH 
#SBATCH -o o%j.ssi2pgapgv
#SBATCH -e o%j.ssi2pgapgv
#SBATCH -J ssi2pgapgv

INFILE=/global/cfs/cdirs/m3354/2021-Summit3600/surface-800-100.ssi
OUTFILE=/global/cfs/cdirs/m3354/2021-Summit3600/surface-800-100-pgapgv-500m-new.h5

date

source activate myenv

stdbuf -o0 -e0 -i0 srun -N 16 -n 256 python /global/u1/h/houhun/sw4essi_converter/ssi2pgapgv.py -i $INFILE -o $OUTFILE -s 500 -v

date
