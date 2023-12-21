#include <iostream>
#include <fstream>
#include <iomanip>
#include "omp.h"
#include <stdio.h>
#include <vector>
#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

const bool is4BodyInput = true;
const bool doOutputToFile = true;

#define blockSize 32
#define TYPE float

const TYPE G = 6.67e-11;
const TYPE eps = 1e-6;

void readFromFile(vector<TYPE>& M, vector<TYPE>& R, vector<TYPE>& V, int& N) {
    std::string filename;
    if (is4BodyInput) {
        filename = "4Body.txt";
    }
    else {
        filename = "10KBody.txt";
    }
    ifstream file;
    file.open(filename);

    file >> N;
    M.resize(N); R.resize(N * 3); V.resize(N * 3);

    for (int i = 0; i < N; ++i) {
        file >> M[i] >> R[3 * i] >> R[3 * i + 1] >> R[3 * i + 2] >> V[3 * i] >> V[3 * i + 1] >> V[3 * i + 2];
    }

    file.close();
}

__global__ void calcAcceleration(TYPE* dev_M, TYPE* dev_R, TYPE* dev_V, TYPE* dev_KV, TYPE* dev_KA, int N) {
    int globIdx = threadIdx.x + blockDim.x * blockIdx.x;
    int locIdx = threadIdx.x;
    int globIdx3 = 3 * globIdx, locIdx3 = 3 * locIdx;

    __shared__ TYPE sharedM[blockSize], sharedR[3 * blockSize];

    TYPE d0, d1, d2, norm, znam, a0 = 0.0, a1 = 0.0, a2 = 0.0;

    TYPE r0 = dev_R[3 * globIdx], r1 = dev_R[3 * globIdx + 1], r2 = dev_R[3 * globIdx + 2];


    for (int i = 0; i < N; i += blockSize) {
        sharedM[locIdx] = dev_M[i + locIdx];
        sharedR[locIdx3] = dev_R[3 * (i + locIdx)];
        sharedR[locIdx3 + 1] = dev_R[3 * (i + locIdx) + 1];
        sharedR[locIdx3 + 2] = dev_R[3 * (i + locIdx) + 2];

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < blockSize; ++j) {
            if (i + j < N) {
                d0 = r0 - sharedR[3 * j];
                d1 = r1 - sharedR[3 * j + 1];
                d2 = r2 - sharedR[3 * j + 2];

                norm = d0 * d0 + d1 * d1 + d2 * d2;

                //norm *= sqrt(norm);
                norm *= __fsqrt_rd(norm);

                //znam = sharedM[j] / fmax(norm, eps);
                znam =  __fdividef(sharedM[j], fmaxf(norm, eps));

                a0 += d0 * znam;
                a1 += d1 * znam;
                a2 += d2 * znam;
            }

        __syncthreads();

        }

    }

    if (globIdx < N) {
        dev_KV[globIdx3] = dev_V[globIdx3];
        dev_KV[globIdx3 + 1] = dev_V[globIdx3 + 1];
        dev_KV[globIdx3 + 2] = dev_V[globIdx3 + 2];

        dev_KA[globIdx3] = -G * a0;
        dev_KA[globIdx3 + 1] = -G * a1;
        dev_KA[globIdx3 + 2] = -G * a2;
    }

}



__global__ void multAdd(TYPE* v0, TYPE* v1, TYPE tau, TYPE* result, int N) {
    int globIdx = threadIdx.x + blockDim.x * blockIdx.x;

    if (globIdx < N) {
        for (int i = 0; i < 3; i++) {
            result[3 * globIdx + i] = v0[3 * globIdx + i] + tau * v1[3 * globIdx + i];
        }
    }
}


void RungeKutta2(const vector<TYPE>& M, vector<TYPE>& R, const vector<TYPE>& V, TYPE tau, TYPE T, int N) {

    int N3 = 3 * N;

    ofstream* F = NULL;


      if (doOutputToFile) {
      F = new ofstream[N];
            for (int i = 0; i < N; ++i) {
                F[i].open(to_string(N) + "_Body_" + to_string(i + 1) + ".txt");
                F[i] << 0. <<  " " << R[3*i + 0] << " " << R[3*i + 1] << " " << R[3*i + 2] << endl;
            }
        }

    TYPE *dev_M, *dev_R, *dev_V, *dev_KV1, *dev_KV2, *dev_KA1, *dev_KA2, *dev_tempR, *dev_tempV;
    TYPE tau2 = tau / 2 ;

    dim3 blocks = ((N + blockSize - 1) / blockSize);
    dim3 threads(blockSize);

    cudaMalloc(&dev_M, N * sizeof(TYPE));
    cudaMalloc(&dev_R, N3 * sizeof(TYPE));
    cudaMalloc(&dev_V, N3 * sizeof(TYPE));
    cudaMalloc(&dev_KV1, N3 * sizeof(TYPE));
    cudaMalloc(&dev_KV2, N3 * sizeof(TYPE));
    cudaMalloc(&dev_KA1, N3 * sizeof(TYPE));
    cudaMalloc(&dev_KA2, N3 * sizeof(TYPE));
    cudaMalloc(&dev_tempR, N3 * sizeof(TYPE));
    cudaMalloc(&dev_tempV, N3 * sizeof(TYPE));

    cudaMemcpy(dev_M, M.data(), N * sizeof(TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_R, R.data(), N3 * sizeof(TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_V, V.data(), N3 * sizeof(TYPE), cudaMemcpyHostToDevice);

    cudaEvent_t start, finish;
    cudaEventCreate(&start);
    cudaEventCreate(&finish);
    cudaEventRecord(start);
    cudaEventSynchronize(start);

    int timesteps = round(T / tau);
    int forPrint = round(1 / (10 * tau));

    for (int i = 1; i <= timesteps; ++i) {
        calcAcceleration <<<blocks, threads>>>(dev_M, dev_R, dev_V, dev_KV1, dev_KA1, N);
        multAdd<<<blocks, threads>>>(dev_R, dev_KV1, tau2, dev_tempR, N);
        multAdd<<<blocks, threads>>>(dev_V, dev_KA1, tau2, dev_tempV, N);

        calcAcceleration <<<blocks, threads>>>(dev_M, dev_tempR, dev_tempV, dev_KV2, dev_KA2, N);
        multAdd<<<blocks, threads>>>(dev_R, dev_KV2, tau, dev_R, N);
        multAdd<<<blocks, threads>>>(dev_V, dev_KA2, tau, dev_V, N);

        if (i % forPrint == 0) {
                        if (doOutputToFile) {
                        TYPE current_time = tau *i;
                     cudaMemcpy(R.data(), dev_R, N3 * sizeof(TYPE), cudaMemcpyDeviceToHost);
                    for (int i = 0; i < N; ++i) {
                        F[i] << current_time << " " << R[3*i+0] << " " << R[3*i+1] << " " << R[3*i+2] << endl;
                    }
                }
        }
    }

    cudaEventRecord(finish);
    cudaEventSynchronize(finish);

    float dt;
    cudaEventElapsedTime(&dt, start, finish);
    cudaEventDestroy(start);
    cudaEventDestroy(finish);


     if (doOutputToFile)
            for (int i = 0; i < N; ++i)
                F[i].close();

    cudaFree(dev_M);
    cudaFree(dev_R);
    cudaFree(dev_V);
    cudaFree(dev_KV1);
    cudaFree(dev_KV2);
    cudaFree(dev_KA1);
    cudaFree(dev_KA2);
    cudaFree(dev_tempR);
    cudaFree(dev_tempV);

    printf("Time = %f\n", dt/1000.0);
}


int main(int argc, char** argv) {
    int N; TYPE T, countstep = 10.0, tau = 1e-3;
    vector<TYPE> rad, mas, vel;
    readFromFile(mas, rad, vel, N);

    if (is4BodyInput) {
        T = 20.0;
    }
    else {
        T = countstep * tau;
    }

    RungeKutta2(mas, rad, vel, tau, T, N);


    return 0;
}
