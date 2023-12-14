#!/bin/sh
#SBATCH --job-name=Sukhova_Pinevich
#SBATCH --time=00:05:00
#SBATCH --nodes=1 --ntasks-per-node=16
#SBATCH --partition release
ulimit -l unlimited
mpirun -n 16 /nethome/student/FS20/FS2-x1/Pinevich_Sukhova/LAB4_test/run
