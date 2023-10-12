#!/bin/sh
#SBATCH --job-name=LAB-1-test
#SBATCH --time=00:01:00
#SBATCH --nodes=1 --cpus-per-task=18
#SBATCH --partition release
ulimit -l unlimited
mpirun -n 3 /nethome/student/FS20/FS2-x1/Pinevich_Sukhova/ParallelComputing/LAB1/executable