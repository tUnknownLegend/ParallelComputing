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

void readFromFile(int &N, vector<TYPE> &weight, vector<TYPE> &position, vector<TYPE> &velocity) {
    string filename;
    if (is4BodyInput) {
        filename = "4Body.txt";
    } else {
        filename = "10KBody.txt";
    }
    ifstream file;
    file.open(filename);

    file >> N;
    weight.resize(N);
    position.resize(N * 3);
    velocity.resize(N * 3);

    for (int i = 0; i < N; ++i) {
        file >> weight[i] >> position[3 * i] >> position[3 * i + 1] >> position[3 * i + 2] >> velocity[3 * i]
             >> velocity[3 * i + 1] >> velocity[3 * i + 2];
    }

    file.close();
}

__global__ void
calcAcceleration(TYPE *cudaWeight, TYPE *cudaPosition, TYPE *cudaVelocity, TYPE *dev_KV, TYPE *dev_KA, int N) {
    int globIdx = threadIdx.x + blockDim.x * blockIdx.x;
    int locIdx = threadIdx.x;
    int globIdx3 = 3 * globIdx, locIdx3 = 3 * locIdx;

    __shared__ TYPE sharedM[blockSize], sharedR[3 * blockSize];

    TYPE d0, d1, d2, norm, znam, a0 = 0.0, a1 = 0.0, a2 = 0.0;

    TYPE r0 = cudaPosition[3 * globIdx], r1 = cudaPosition[3 * globIdx + 1], r2 = cudaPosition[3 * globIdx + 2];


    for (int i = 0; i < N; i += blockSize) {
        sharedM[locIdx] = cudaWeight[i + locIdx];
        sharedR[locIdx3] = cudaPosition[3 * (i + locIdx)];
        sharedR[locIdx3 + 1] = cudaPosition[3 * (i + locIdx) + 1];
        sharedR[locIdx3 + 2] = cudaPosition[3 * (i + locIdx) + 2];

        __syncthreads();

        for (int j = 0; j < blockSize; ++j) {
            if (i + j < N) {
                d0 = r0 - sharedR[3 * j];
                d1 = r1 - sharedR[3 * j + 1];
                d2 = r2 - sharedR[3 * j + 2];

                norm = d0 * d0 + d1 * d1 + d2 * d2;

                //norm *= sqrt(norm);
                norm *= __fsqrt_rd(norm);

                //znam = sharedM[j] / fmax(norm, eps);
                znam = __fdividef(sharedM[j], fmaxf(norm, eps));

                a0 += d0 * znam;
                a1 += d1 * znam;
                a2 += d2 * znam;
            }

            __syncthreads();

        }

    }

    if (globIdx < N) {
        dev_KV[globIdx3] = cudaVelocity[globIdx3];
        dev_KV[globIdx3 + 1] = cudaVelocity[globIdx3 + 1];
        dev_KV[globIdx3 + 2] = cudaVelocity[globIdx3 + 2];

        dev_KA[globIdx3] = -G * a0;
        dev_KA[globIdx3 + 1] = -G * a1;
        dev_KA[globIdx3 + 2] = -G * a2;
    }

}


__global__ void multAdd(TYPE *v0, TYPE *v1, TYPE tau, TYPE *result, int N) {
    int globIdx = threadIdx.x + blockDim.x * blockIdx.x;

    if (globIdx < N) {
        for (int i = 0; i < 3; i++) {
            result[3 * globIdx + i] = v0[3 * globIdx + i] + tau * v1[3 * globIdx + i];
        }
    }
}


void
RungeKutta2(const vector<TYPE> &weight, vector<TYPE> &position, const vector<TYPE> &velocity, TYPE tau, TYPE T, int N) {
    const int N3 = 3 * N;
    ofstream *F = nullptr;

    if (doOutputToFile) {
        F = new ofstream[N];
        for (int i = 0; i < N; ++i) {
            F[i].open(to_string(N) + "_Body_" + to_string(i + 1) + ".txt");
            F[i] << 0. << " " << position[3 * i + 0] << " " << position[3 * i + 1] << " " << position[3 * i + 2]
                 << endl;
        }
    }

    TYPE *cudaWeight;
    cudaMalloc(&cudaWeight, N * sizeof(TYPE));

    TYPE *cudaPosition;
    cudaMalloc(&cudaPosition, N3 * sizeof(TYPE));

    TYPE *cudaVelocity;
    cudaMalloc(&cudaVelocity, N3 * sizeof(TYPE));

    TYPE *dev_KV1;
    cudaMalloc(&dev_KV1, N3 * sizeof(TYPE));

    TYPE *dev_KV2;
    cudaMalloc(&dev_KV2, N3 * sizeof(TYPE));

    TYPE *dev_KA1;
    cudaMalloc(&dev_KA1, N3 * sizeof(TYPE));

    TYPE *dev_KA2;
    cudaMalloc(&dev_KA2, N3 * sizeof(TYPE));

    TYPE *tempCudaPosition;
    cudaMalloc(&tempCudaPosition, N3 * sizeof(TYPE));

    TYPE *tempCudaVelocity;
    cudaMalloc(&tempCudaVelocity, N3 * sizeof(TYPE));

    cudaMemcpy(cudaWeight, weight.data(), N * sizeof(TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaPosition, position.data(), N3 * sizeof(TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaVelocity, velocity.data(), N3 * sizeof(TYPE), cudaMemcpyHostToDevice);

    cudaEvent_t start;
    cudaEvent_t finish;
    cudaEventCreate(&start);
    cudaEventCreate(&finish);
    cudaEventRecord(start);
    cudaEventSynchronize(start);

    const int timeSteps = round(T / tau);
    const int forPrint = round(1 / (10 * tau));

    TYPE halfOfTau = tau / 2;
    dim3 blocks = ((N + blockSize - 1) / blockSize);
    dim3 threads(blockSize);

    for (int i = 1; i <= timeSteps; ++i) {
        calcAcceleration <<<blocks, threads>>>(cudaWeight, cudaPosition, cudaVelocity, dev_KV1, dev_KA1, N);
        multAdd<<<blocks, threads>>>(cudaPosition, dev_KV1, halfOfTau, tempCudaPosition, N);
        multAdd<<<blocks, threads>>>(cudaVelocity, dev_KA1, halfOfTau, tempCudaVelocity, N);

        calcAcceleration <<<blocks, threads>>>(cudaWeight, tempCudaPosition, tempCudaVelocity, dev_KV2, dev_KA2, N);
        multAdd<<<blocks, threads>>>(cudaPosition, dev_KV2, tau, cudaPosition, N);
        multAdd<<<blocks, threads>>>(cudaVelocity, dev_KA2, tau, cudaVelocity, N);

        if (i % forPrint == 0) {
            if (doOutputToFile) {
                TYPE current_time = tau * i;
                cudaMemcpy(position.data(), cudaPosition, N3 * sizeof(TYPE), cudaMemcpyDeviceToHost);
                for (int i = 0; i < N; ++i) {
                    F[i] << current_time << " " << position[3 * i + 0] << " " << position[3 * i + 1] << " "
                         << position[3 * i + 2] << endl;
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


    if (doOutputToFile) {
        for (int i = 0; i < N; ++i) {
            F[i].close();
        }
    }

    cudaFree(cudaWeight);
    cudaFree(cudaPosition);
    cudaFree(cudaVelocity);
    cudaFree(dev_KV1);
    cudaFree(dev_KV2);
    cudaFree(dev_KA1);
    cudaFree(dev_KA2);
    cudaFree(tempCudaPosition);
    cudaFree(tempCudaVelocity);

    printf("Time = %f\n", dt / 1000.0);
}


int main() {
    int N;
    TYPE T;
    TYPE countStep = 10.0;
    TYPE tau = 1e-3;
    vector<TYPE> weight;
    vector<TYPE> position;
    vector<TYPE> velocity;
    readFromFile(N, weight, position, velocity);

    if (is4BodyInput) {
        T = 20.0;
    } else {
        T = countStep * tau;
    }

    RungeKutta2(mas, rad, vel, tau, T, N);

    return 0;
}
