#!/bin/sh
#SBATCH --job-name=LAB-2-test
#SBATCH --time=00:05:00
#SBATCH --nodes=4 --cpus-per-task=1
#SBATCH --partition release
ulimit -l unlimited
mpirun -n 4 /nethome/student/FS20/FS2-x1/Pinevich_Sukhova/LAB2/executable
