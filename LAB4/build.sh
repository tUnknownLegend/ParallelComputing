nvcc -O2 -gencode arch=compute_70,code=sm_70 -o run kernel.cu -Xcompiler -fopenmp
