nvcc -O3 -gencode arch=compute_70,code=sm_70 -o exec kernel.cu -Xcompiler -fopenmp
