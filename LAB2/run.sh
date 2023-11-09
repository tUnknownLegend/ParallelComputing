#!/bin/sh
#SBATCH --job-name=LAB-2-test
#SBATCH --time=00:15:00
#SBATCH --nodes=1 --cpus-per-task=18
#SBATCH --partition release
ulimit -l unlimited
mpirun -n 1 /nethome/student/FS20/FS2-x1/Pinevich_Sukhova/LAB2/executable
