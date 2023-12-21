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

bool is4BodyInput = true;
bool doOutput = true;


#define blockSize 128
#define TYPE double

const TYPE G = 6.67e-11;
const TYPE eps = 1e-6;

void readFromFile(vector<TYPE> &weight, vector<TYPE> &position, vector<TYPE> &velocity, int &N) {
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
acceleration(TYPE *cudaWeight, TYPE *cudaPosition, TYPE *cudaVelocity, TYPE *dev_KV, TYPE *dev_KA, int N) {
    int globIdx = threadIdx.x + blockDim.x * blockIdx.x;
    int locIdx = threadIdx.x;
    int globIdx3 = 3 * globIdx;
    int locIdx3 = 3 * locIdx;

    __shared__ TYPE sharedWeight[blockSize], sharedPosition[3 * blockSize];

    TYPE diff0, diff1, diff2, norm, mul, a0 = 0.0, a1 = 0.0, a2 = 0.0;

    TYPE position0 = cudaPosition[globIdx3], position1 = cudaPosition[globIdx3 + 1], position2 = cudaPosition[globIdx3 +
                                                                                                              2];

    for (int i = 0; i < N; i += blockSize) {
        sharedWeight[locIdx] = cudaWeight[i + locIdx];
        sharedPosition[locIdx3] = cudaPosition[3 * (i + locIdx)];
        sharedPosition[locIdx3 + 1] = cudaPosition[3 * (i + locIdx) + 1];
        sharedPosition[locIdx3 + 2] = cudaPosition[3 * (i + locIdx) + 2];

        __syncthreads();

#pragma unroll
        for (int j = 0; j < N - i; ++j) {
            diff0 = position0 - sharedPosition[3 * j];
            diff1 = position1 - sharedPosition[3 * j + 1];
            diff2 = position2 - sharedPosition[3 * j + 2];

            norm = diff0 * diff0 + diff1 * diff1 + diff2 * diff2;

            mul = sharedWeight[j] / fmax(norm * __fsqrt_rn(norm), eps);
            //mul =  __fdividef(sharedWeight[j], fmax(norm * __fsqrt_rn(norm), eps));

            a0 += diff0 * mul;
            a1 += diff1 * mul;
            a2 += diff2 * mul;

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
    ofstream *F = nullptr;
    if (doOutput) {
        F = new ofstream[N];
        for (int i = 0; i < N; ++i) {
            F[i].open(to_string(num) + "_Body_" + to_string(i + 1) + ".txt");
            F[i] << 0. << " " << R[3 * i + 0] << " " << R[3 * i + 1] << " " << R[3 * i + 2] << endl;
        }
    }

    TYPE halfOfTau = tau / 2; // half of step

    dim3 blocks = ((N + blockSize - 1) / blockSize);
    dim3 threads(blockSize);

    TYPE *cudaWeight;
    cudaMalloc(&cudaWeight, N * sizeof(TYPE));

    TYPE *cudaPosition;
    cudaMalloc(&cudaPosition, N3 * sizeof(TYPE));

    TYPE *cudaVelocity;
    cudaMalloc(&cudaVelocity, N3 * sizeof(TYPE));

    TYPE *cudaKVelocity1;
    cudaMalloc(&cudaKVelocity1, N3 * sizeof(TYPE));

    TYPE *cudaKVelocity2;
    cudaMalloc(&cudaKVelocity2, N3 * sizeof(TYPE));

    TYPE *dev_KA1;
    cudaMalloc(&dev_KA1, N3 * sizeof(TYPE));

    TYPE *dev_KA2;
    cudaMalloc(&dev_KA2, N3 * sizeof(TYPE));

    TYPE *cudaTempPosition;
    cudaMalloc(&cudaTempPosition, N3 * sizeof(TYPE));

    TYPE *cudaTempVelocity;
    cudaMalloc(&cudaTempVelocity, N3 * sizeof(TYPE));

    cudaMemcpy(cudaWeight, M.data(), N * sizeof(TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaPosition, R.data(), N3 * sizeof(TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaVelocity, V.data(), N3 * sizeof(TYPE), cudaMemcpyHostToDevice);

    cudaEvent_t start, finish;
    cudaEventCreate(&start);
    cudaEventCreate(&finish);
    cudaEventRecord(start);
    cudaEventSynchronize(start);

    int timeSteps = round(T / tau);
    int forPrint = round(1 / (10 * tau));

    for (int i = 1; i <= timeSteps; ++i) {
        acceleration <<<blocks, threads>>>(cudaWeight, cudaPosition, cudaVelocity, cudaKVelocity1, dev_KA1, N);
        RKstep<<<blocks, threads>>>(cudaPosition, cudaKVelocity1, halfOfTau, cudaTempPosition, N);
        RKstep<<<blocks, threads>>>(cudaVelocity, dev_KA1, halfOfTau, cudaTempVelocity, N);

        acceleration <<<blocks, threads>>>(cudaWeight, cudaTempPosition, cudaTempVelocity, cudaKVelocity2, dev_KA2, N);
        RKstep<<<blocks, threads>>>(cudaPosition, cudaKVelocity2, tau, cudaPosition, N);
        RKstep<<<blocks, threads>>>(cudaVelocity, dev_KA2, tau, cudaVelocity, N);

        if (doOutput && i % forPrint == 0) {
            TYPE current_time = tau * i;
            cudaMemcpy(R.data(), cudaPosition, N3 * sizeof(TYPE), cudaMemcpyDeviceToHost);
            for (int i = 0; i < N; ++i) {
                F[i] << current_time << " " << std::setprecision(16) << R[3 * i + 0] << " " << R[3 * i + 1] << " "
                     << R[3 * i + 2] << endl;
            }
        }
    }

    cudaEventRecord(finish);
    cudaEventSynchronize(finish);

    float dt;
    cudaEventElapsedTime(&dt, start, finish);
    cudaEventDestroy(start);
    cudaEventDestroy(finish);


    if (doOutput) {
        for (int i = 0; i < N; ++i) {
            F[i].close();
        }
    }

    cudaFree(cudaWeight);
    cudaFree(cudaPosition);
    cudaFree(cudaVelocity);
    cudaFree(cudaKVelocity1);
    cudaFree(cudaKVelocity2);
    cudaFree(dev_KA1);
    cudaFree(dev_KA2);
    cudaFree(cudaTempPosition);
    cudaFree(cudaTempVelocity);

    printf("Time = %f\n", dt / (timeSteps * 1000.0));
}

int main() {
    int N;
    TYPE T;
    TYPE countStep = 10.0;
    const int num = 4;
    TYPE tau = num * 1e-3; // step
    vector<TYPE> weight;
    vector<TYPE> position;
    vector<TYPE> velocity;
    readFromFile(weight, position, velocity, N);

    if (is4BodyInput) {
        T = 20.0;
    } else {
        T = countStep * tau;
    }

    RK2(weight, position, velocity, tau, T, N, num);

    return 0;
}
