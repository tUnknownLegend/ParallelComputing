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

bool f1 = true;
bool f2 = true;


#define blocksize 32
#define TYPE double

const TYPE G = 6.67e-11;
const TYPE eps = 1e-6;

void readFromFile(vector<TYPE> &M, vector<TYPE> &R, vector<TYPE> &V, int &N) {
    std::string filename;
    if (f1) {
        filename = "4Body.txt";
    } else {
        filename = "10KBody.txt";
    }
    ifstream file;
    file.open(filename);

    file >> N;
    M.resize(N);
    R.resize(N * 3);
    V.resize(N * 3);

    for (int i = 0; i < N; ++i) {
        file >> M[i] >> R[3 * i] >> R[3 * i + 1] >> R[3 * i + 2] >> V[3 * i] >> V[3 * i + 1] >> V[3 * i + 2];
    }

    file.close();
}

__global__ void acceleration(TYPE *dev_M, TYPE *dev_R, TYPE *dev_V, TYPE *dev_KV, TYPE *dev_KA, int N) {
    int globIdx = threadIdx.x + blockDim.x * blockIdx.x;
    int locIdx = threadIdx.x;
    int globIdx3 = 3 * globIdx;
    int locIdx3 = 3 * locIdx;

    __shared__ TYPE sharedM[blocksize], sharedR[3 * blocksize];

    TYPE diff0, diff1, diff2, norm, mul, a0 = 0.0, a1 = 0.0, a2 = 0.0;

    TYPE r0 = dev_R[globIdx3], r1 = dev_R[globIdx3 + 1], r2 = dev_R[globIdx3 + 2];


    for (int i = 0; i < N; i += blocksize) {
        sharedM[locIdx] = dev_M[i + locIdx];
        sharedR[locIdx3] = dev_R[3 * (i + locIdx)];
        sharedR[locIdx3 + 1] = dev_R[3 * (i + locIdx) + 1];
        sharedR[locIdx3 + 2] = dev_R[3 * (i + locIdx) + 2];

        __syncthreads();

#pragma unroll
        for (int j = 0; j < N - i; ++j) {
            diff0 = r0 - sharedR[3 * j];
            diff1 = r1 - sharedR[3 * j + 1];
            diff2 = r2 - sharedR[3 * j + 2];

            norm = diff0 * diff0 + diff1 * diff1 + diff2 * diff2;

            norm *= sqrt(norm);
            //norm *= __fsqrt_rn(norm);

            mul = sharedM[j] / fmax(norm, eps);
            //mul =  __fdividef(sharedM[j], fmaxf(norm, eps));

            a0 += diff0 * mul;
            a1 += diff1 * mul;
            a2 += diff2 * mul;

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


__global__ void RKstep(TYPE *y0, TYPE *y1, TYPE tau, TYPE *result, int N) {
    int globIdx = threadIdx.x + blockDim.x * blockIdx.x;
    int globIdx3 = 3 * globIdx;
    if (globIdx < N) {
        for (int i = 0; i < 3; i++) {
            result[globIdx3 + i] = y0[globIdx3 + i] + tau * y1[globIdx3 + i];
        }
    }
}


void RK2(const vector<TYPE> &M, vector<TYPE> &R, const vector<TYPE> &V, TYPE tau, TYPE T, int N, int num) {

    int N3 = 3 * N;

    ofstream *F = NULL;


    if (f2) {
        F = new ofstream[N];
        for (int i = 0; i < N; ++i) {
            F[i].open(to_string(num) + "_Body_" + to_string(i + 1) + ".txt");
            F[i] << 0. << " " << R[3 * i + 0] << " " << R[3 * i + 1] << " " << R[3 * i + 2] << endl;
        }
    }

    TYPE *dev_M, *dev_R, *dev_V, *dev_KV1, *dev_KV2, *dev_KA1, *dev_KA2, *dev_tempR, *dev_tempV;
    TYPE tau2 = tau / 2;

    dim3 blocks = ((N + blocksize - 1) / blocksize);
    dim3 threads(blocksize);

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
        acceleration <<<blocks, threads>>>(dev_M, dev_R, dev_V, dev_KV1, dev_KA1, N);
        RKstep<<<blocks, threads>>>(dev_R, dev_KV1, tau2, dev_tempR, N);
        RKstep<<<blocks, threads>>>(dev_V, dev_KA1, tau2, dev_tempV, N);

        acceleration <<<blocks, threads>>>(dev_M, dev_tempR, dev_tempV, dev_KV2, dev_KA2, N);
        RKstep<<<blocks, threads>>>(dev_R, dev_KV2, tau, dev_R, N);
        RKstep<<<blocks, threads>>>(dev_V, dev_KA2, tau, dev_V, N);

        if (i % forPrint == 0) {
            if (f2) {
                TYPE current_time = tau * i;
                cudaMemcpy(R.data(), dev_R, N3 * sizeof(TYPE), cudaMemcpyDeviceToHost);
                for (int i = 0; i < N; ++i) {
                    F[i] << current_time << " " << std::setprecision(16) << R[3 * i + 0] << " " << R[3 * i + 1] << " "
                         << R[3 * i + 2] << endl;
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


    if (f2)
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

    printf("Time = %f\n", dt / (timesteps * 1000.0));
    //printf("ALlTime = %f\n", dt/1000.0);
}

int main() {
    int N;
    TYPE T;
    TYPE countStep = 10.0;
    TYPE tau = 1e-3; // step
    vector<TYPE> weight;
    vector<TYPE> position;
    vector<TYPE> velocity;
    readFromFile(N, weight, position, velocity);

    if (is4BodyInput) {
        T = 20.0;
    } else {
        T = countStep * tau;
    }

    RungeKutta2(weight, position, velocity, tau, T, N);

    return 0;
}
