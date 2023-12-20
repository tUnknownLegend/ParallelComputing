#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <iomanip>
#include <omp.h>

#define MyType float

const int blockS = 256;

struct Body // Структура "тело"
{
    MyType weight;    // Масса
    MyType position[3]; // Координаты
    MyType velocity[3]; // Скорости

    // Перегрузка оператора присваивания
    __device__ __host__ Body &operator=(const Body &p) {
        weight = p.weight;
        for (int i = 0; i < 3; ++i) {
            position[i] = p.position[i];
            velocity[i] = p.velocity[i];
        }
        return *this;
    }
};

// Перегрузка оператора вывода для структуры "тело"
std::ostream &operator<<(std::ostream &str, const Body &b) {
    str << std::setprecision(10) << b.position[0] << " " << b.position[1] << " " << b.position[2] << std::endl;

    return str;
}

void WriteFile(const std::string &file, const MyType position[3], MyType t, int glob_i) {
    std::ofstream F(file + std::to_string(glob_i) + ".txt", std::ios::app);
    F << std::setprecision(10) << t << " " << position[0] << " " << position[1] << " " << position[2] << std::endl;
    F.close();
    F.clear();
}

// Модуль вектора
__device__ inline MyType My_norm_vec(const MyType *position) {
    return position[0] * position[0] + position[1] * position[1] + position[2] * position[2];
}

// Вычисление ускорения
__device__ void My_a(MyType *a, const size_t N, const MyType *position, const int glob_i, const Body *data, MyType G) {
    MyType buf[3] = {0.0, 0.0, 0.0};

    for (size_t k = 0; k < 3; ++k)
        a[k] = 0.0;

    MyType coefN = 1.0;

    Body bod_j;

    float4 dob4;

    __shared__ float4 SharedBlock[512];

    for (int k = 0; k < N / blockDim.x; ++k) {
        bod_j = data[blockDim.x * k + threadIdx.x];

        SharedBlock[threadIdx.x] = make_float4(bod_j.position[0], bod_j.position[1], bod_j.position[2], bod_j.weight);

        __syncthreads();

        for (size_t j = 0; j < blockDim.x; ++j) {
            if (glob_i == blockDim.x * k + j)
                continue;

            dob4 = SharedBlock[j];

            // for (size_t k = 0; k < 3; ++k)
            // buf[k] = bod_j[k] - position[k];

            buf[0] = dob4.x - position[0];
            buf[1] = dob4.y - position[1];
            buf[2] = dob4.z - position[2];

            coefN = buf[0] * buf[0] + buf[1] * buf[1] + buf[2] * buf[2]; //My_norm_vec(buf);

            //coefN *= sqrtf(coefN);

            // coefN *= coefN * coefN;

            coefN = __fdividef(rsqrtf(coefN), coefN) * dob4.w;

#pragma unroll
            for (size_t k = 0; k < 3; ++k) {
                a[k] += coefN * buf[k];
            }


        }

        __syncthreads();
    }

#pragma unroll
    for (size_t k = 0; k < 3; ++k) {
        a[k] *= G;
    }
}

//  This function generates a random double in [i, j]
double GetRandomDouble(double i, double j) {
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(i, j);
    return dis(gen);
}

__global__  void simulate(int N, Body *data, MyType tau, int flagF) {
    MyType timeStep = 1e-1;  // Шаг записи в файлы

    MyType tn = 20.0; // Конечный момент времени

    size_t Nt = round(tn / tau);       // Количество шагов по времени
    size_t tf = round(timeStep / tau); // Коэффициент пропорциональности шагов

    const MyType G = 6.67e-11; // Гравитационная постоянная

    int glob_i = blockIdx.x * blockDim.x + threadIdx.x; // Текущий номер

    //std::ofstream F("Body_" + std::to_string(glob_i) + ".txt"); // Файл выходных данных
    //std::ofstream F("Body.txt", std::ios::app); // Файл выходных данных

    Body bod_i = data[glob_i]; // Текущее тело

    //F << 0.0 << " " << bod_i;

    MyType a[3] = {0.0, 0.0, 0.0}; // Текущие ускорения
    MyType w[3] = {0.0, 0.0, 0.0}; // Начальные ускорения

    Body buf;

    // Body* globData = new Body[N];

    // Расчётная схема

    if (glob_i < N)

        for (size_t t = 1; t <= Nt; ++t) {
            buf = bod_i;
            My_a(w, N, bod_i.position, glob_i, data, G);

            for (size_t k = 0; k < 3; ++k)
                buf.position[k] += tau * bod_i.velocity[k];

            data[glob_i] = buf;

            __syncthreads();

            My_a(a, N, buf.position, glob_i, data, G);

            for (size_t k = 0; k < 3; ++k) {
                bod_i.position[k] += tau * (bod_i.velocity[k] + 0.5 * tau * w[k]);
                bod_i.velocity[k] += 0.5 * tau * (w[k] + a[k]);
            }

            data[glob_i] = bod_i;

            __syncthreads();

            //if (t % tf == 0 && flagF)
            // F << t * tau << " " << bod_i;
        }

    //F.close();
}

int main(int argc, char **argv) {
    size_t block = 512; // Размер блока

    size_t N = 4; // Количество тел

    MyType tau = 1e-1; // Шаг по времени

    int flagInitData = 1; // != 0 - считывать из файла, 0 - заполнять случайно
    int flagF = 1;        // != 0 - записывать в файлы, 0 - нет

    // границы значения масс
    const pair<MyType, MyType> weightBounds{1e+9, 1e+10};

    // границы значения координат
    const pair<MyType, MyType> positionBounds{1e+9, 1e+10};

    // границы значения скоростей
    const pair<MyType, MyType> velocityBounds{1e+9, 1e+10};

    Body *data; // Массив "тел"

    if (flagInitData) {
        std::ifstream F("Input.txt"); // Файл входных данных

        F >> N; // Количество тел

        data = new Body[N]; // Массив "тел"

        for (size_t i = 0; i < N; ++i)
            F >> data[i].weight >> data[i].position[0] >> data[i].position[1] >> data[i].position[2]
              >> data[i].velocity[0] >> data[i].velocity[1]
              >> data[i].velocity[2];

        F.close();
    } else {
        data = new Body[N]; // Массив "тел"

        for (size_t i = 0; i < N; ++i) {

            data[i].weight = GetRandomDouble(weightBounds.first, weightBounds.second);

            for (size_t k = 0; k < 3; ++k) {
                data[i].position[k] = GetRandomDouble(positionBounds.first, positionBounds.second);
                data[i].velocity[k] = GetRandomDouble(velocityBounds.first, velocityBounds.second);
            }
        }
    }

    std::cout << "N = " << N << std::endl;
    std::cout << "tau = " << tau << std::endl;

    Body *GPUdata;

    // Копирование массива data в видеокарты
    cudaMalloc((void **) &GPUdata, N * sizeof(Body));
    cudaMemcpy(GPUdata, data, N * sizeof(Body), cudaMemcpyHostToDevice);

    // Обработчик событий
    cudaEvent_t start, stop;
    float time = 0.0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    size_t blockCount = 0;

    if (N < block) {
        simulate<<<1, N>>>(N, GPUdata, tau, flagF);
    } else {
        blockCount = N / block + N % block;
        simulate<<<blockCount, block>>>(N, GPUdata, tau, flagF);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "time = " << time / 1000.0 << std::endl << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(data, GPUdata, N * sizeof(Body), cudaMemcpyHostToDevice);

    cudaFree(GPUdata);
    delete[]data;

    return 0;
}