#include <iostream>
#include <fstream>
#include <cstring>
#include <random>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define mytype float

using namespace std;

int N = 4;

bool NeedWriteToFile = false;
bool ManyBody = false;

const string input_file_name = "Input.txt";
const string gendata_file_name = "genbody.txt";
const string output_file_name = "traj.txt";

const mytype G = 6.67e-11;
const mytype eps = 1e-3;
mytype T = 20.0;
mytype tau = 0.01;
const double output_tau = 0.1;
const int stop_iter = 10;
const int block_size = 128;

__device__ mytype norm_minus(mytype x, mytype y, mytype z);

void ReadFromFile(const string& filename, vector<mytype>& mass, vector <mytype>& coord, vector<mytype>& vel);
__host__ void GenDataInFile (int N, const string filename);
__host__ void WriteToFile(const string &file_name, mytype t, int num_body, mytype x1, mytype x2, mytype x3);
__host__ void ClearFile(const std::string &file_name);

__global__ void AccVel(int N, mytype* device_mass, mytype* device_coord, mytype* device_vel, mytype* k_coord, mytype* k_vel);
__global__ void FindSol(int N, mytype* prev, mytype* next, mytype h, mytype* res);
__global__ void FinalStep(int N, mytype* device_coord, mytype* device_vel, mytype tau, mytype* k2_coord, mytype* k2_vel);



int main(int argc, char** argv) {

    vector <mytype> mass;
    vector <mytype> coord;
    vector <mytype> vel;

    if (!ManyBody) {
        ReadFromFile(input_file_name, mass, coord, vel);
    }
    else {
        GenDataInFile(N, gendata_file_name);
        ReadFromFile(gendata_file_name, mass, coord, vel);
    }

    int size = mass.size();
    int size3 = 3 * size;

    if (!NeedWriteToFile)
        T = tau * mytype(stop_iter);

    mytype* device_mass;
    mytype* device_coord;
    mytype* device_vel;

    cudaMalloc(&device_mass, size * sizeof(mytype));
    cudaMalloc(&device_coord, size3 * sizeof(mytype));
    cudaMalloc(&device_vel, size3 * sizeof(mytype));

    mytype* tempDevice_coord;
    mytype* tempDevice_vel;
    mytype* k1_coord;
    mytype* k2_coord;
    mytype* k1_vel;
    mytype* k2_vel;

    cudaMalloc(&tempDevice_coord, size3 * sizeof(mytype));
    cudaMalloc(&tempDevice_vel, size3 * sizeof(mytype));
    cudaMalloc(&k1_coord, size3 * sizeof(mytype));
    cudaMalloc(&k2_coord, size3 * sizeof(mytype));
    cudaMalloc(&k1_vel, size3 * sizeof(mytype));
    cudaMalloc(&k2_vel, size3 * sizeof(mytype));

    cudaMemcpy(device_mass, mass.data(), size * sizeof(mytype), cudaMemcpyHostToDevice);
    cudaMemcpy(device_coord, coord.data(), size3 * sizeof(mytype), cudaMemcpyHostToDevice);
    cudaMemcpy(device_vel, vel.data(), size3 * sizeof(mytype), cudaMemcpyHostToDevice);

    dim3 blocks = dim3(N / block_size + (N % block_size != 0) , 1, 1);
    dim3 threads = dim3(block_size, 1, 1);

    mytype half_tau = tau / 2;
    double t = 0.0;

    // if (NeedWriteToFile) {
    // for (int i = 0; i < size; ++i) {
    // ClearFile(to_string(i + 1) + output_file_name);
    // WriteToFile(output_file_name, t, i, coord[3 * i], coord[3 * i + 1], coord[3 * i + 2]);
    // }

    // }
    double temp = 1.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);

    int iter = 0;

//Рунге-Кутты-2

    while (t < T + half_tau)
    {
        AccVel <<<blocks, threads >>> (size, device_mass, device_coord, device_vel, k1_coord, k1_vel);

        FindSol <<<blocks, threads >>> (size, device_coord, k1_coord, half_tau, tempDevice_coord);
        FindSol <<<blocks, threads >>> (size, device_vel, k1_vel, half_tau, tempDevice_vel);

        AccVel <<<blocks, threads >>> (size, device_mass, tempDevice_coord, tempDevice_vel, k2_coord, k2_vel);

        FinalStep <<<blocks, threads >>> (size, device_coord, device_vel, tau, k2_coord, k2_vel);

        iter += 1;
        t += tau;

        if (NeedWriteToFile) {
            if (fabs(t - temp * output_tau) < 1e-5) {
                cudaMemcpy(coord.data(), device_coord, size3 * sizeof(mytype), cudaMemcpyDeviceToHost);


                temp += 1.0;
            }
        }

    }

    for (int i = 0; i < size; ++i) {
        WriteToFile(output_file_name, T + half_tau, i, coord[3 * i], coord[3 * i + 1], coord[3 * i + 2]);
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float time;
    cudaEventElapsedTime(&time,start,stop);
    printf("Iterations: %d\n", iter);
    printf("Time spent by GPU on 1 iteration: %f\n", (time/1000.0)/iter);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    cudaFree(device_coord);
    cudaFree(device_vel);
    cudaFree(device_mass);
    cudaFree(tempDevice_coord);
    cudaFree(tempDevice_vel);
    cudaFree(k1_coord);
    cudaFree(k2_coord);
    cudaFree(k1_vel);
    cudaFree(k2_vel);

    return 0;
}



void ReadFromFile(const string& filename, vector<mytype>& mass, vector <mytype>& coord, vector<mytype>& vel) {
    ifstream inf(filename);
    if (!inf) {
        cerr << "File not found :(" << endl;
        exit(1);
    }
    inf >> N;

    mass.resize(N);
    coord.resize(N * 3);
    vel.resize(N * 3);

    for (int i = 0; i < N; i++) {
        inf >> mass[i] >> coord[3 * i]  >> coord[3 * i + 1]  >> coord[3 * i + 2] \
        >> vel[3 * i] >> vel[3 * i + 1] >> vel[3 * i + 2];
    }
    inf.close();
}

__host__ void GenDataInFile (int N, const string filename) {
    ofstream out(filename);
    if (!out) {
        cerr << "File not found :(" << endl;
        exit(1);
    }
    out << N << "\n";

    mytype buf = 0.0;

    for (int i=0; i<N; i++) {
        random_device random_device;
        mt19937 gen(random_device());
        uniform_int_distribution <> dist_m(1e6, 1e8);
        uniform_int_distribution <> dist_v_r(-10, 10);

        //mass
        buf = dist_m(gen);
        out << buf;

        //3 для coord + 3 для vel
        for (int j=0; j < 6; j++) {
            buf = dist_v_r(gen);
            out << " " << buf;
        }
        out << "\n" ;
    }
    out.close();
}
__device__ mytype norm_minus(mytype x, mytype y, mytype z)
{
    mytype norm = (x * x) + (y * y) + (z * z);
    return sqrt(norm);
}

__device__ inline mytype sqr(mytype x)
{return x*x;}

__device__ inline mytype cub(mytype x)
{return x*x*x;}


__global__ void AccVel(int N, mytype* device_mass, mytype* device_coord, mytype* device_vel, mytype* k_coord, mytype* k_vel) {
    int GlobalIdx = threadIdx.x + blockDim.x * blockIdx.x;
    int GlobalIdx3 = 3 * (threadIdx.x + blockDim.x * blockIdx.x);
    int LocalIdx = threadIdx.x;
    int LocalIdx3 = 3 * LocalIdx;

    __shared__ mytype shared_coord[3 * block_size];
    __shared__ mytype shared_mass[block_size];

    mytype eps2 = eps*eps;

    mytype k_vel1 = 0.0;
    mytype k_vel2 = 0.0;
    mytype k_vel3 = 0.0;
    mytype a;
    mytype my_coord1 = device_coord[GlobalIdx3];
    mytype my_coord2 = device_coord[GlobalIdx3 + 1];
    mytype my_coord3 = device_coord[GlobalIdx3 + 2];
    mytype diff_coord1;
    mytype diff_coord2;
    mytype diff_coord3;

    mytype denom;
    mytype d2;
    for (int i = 0; i < N; i += block_size) {
        shared_mass[LocalIdx] = device_mass[i + LocalIdx];
        shared_coord[LocalIdx3]     = device_coord[3 * (i + LocalIdx)];
        shared_coord[LocalIdx3 + 1] = device_coord[3 * (i + LocalIdx) + 1];
        shared_coord[LocalIdx3 + 2] = device_coord[3 * (i + LocalIdx) + 2];

        __syncthreads();

        for (int j = 0; j < block_size; ++j) {
            if (i + j < N) {
                diff_coord1 = my_coord1 - shared_coord[3 * j];
                diff_coord2 = my_coord2 - shared_coord[3 * j + 1];
                diff_coord3 = my_coord3 - shared_coord[3 * j + 2];

                d2 = fmaxf(sqr(diff_coord1)+sqr(diff_coord2)+sqr(diff_coord3), eps2);

                a = __fdividef(shared_mass[j] * rsqrtf(d2), d2);

                //denom = norm_minus(diff_coord1, diff_coord2, diff_coord3) * norm_minus(diff_coord1, diff_coord2, diff_coord3) * norm_minus(diff_coord1, diff_coord2, diff_coord3);
                //a = shared_mass[j] / max(denom, eps3);

                k_vel1 += diff_coord1 * a;
                k_vel2 += diff_coord2 * a;
                k_vel3 += diff_coord3 * a;
            }
        }
        __syncthreads();
    }
    if (GlobalIdx < N) {
        k_vel[GlobalIdx3]     = -G * k_vel1;
        k_vel[GlobalIdx3 + 1] = -G * k_vel2;
        k_vel[GlobalIdx3 + 2] = -G * k_vel3;

        for (size_t i = 0; i < 3; ++i) {
            k_coord[GlobalIdx3 + i] = device_vel[GlobalIdx3 + i];
        }
    }
}

__global__ void FindSol(int N, mytype* prev, mytype* next, mytype h, mytype* res) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < N) {
        for (int i = 0; i < 3; i++) {
            res[3 * idx + i] = prev[3 * idx + i] + h * next[3 * idx + i];
        }
    }
}

__global__ void FinalStep(int N, mytype* device_coord, mytype* device_vel, mytype tau, mytype* k2_coord, mytype* k2_vel) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        for (int i = 0; i < 3; i++) {
            device_coord[3 * idx + i] += tau * k2_coord[3 * idx + i];
            device_vel[3 * idx + i] += tau * k2_vel[3 * idx + i];
        }
    }
}

__host__ void WriteToFile(const string &file_name, mytype t, int num_body, mytype x1, mytype x2, mytype x3) {
    ofstream file(to_string(num_body + 1) + file_name, std::ios::app);
    file << t << "    ";
    file << x1 << "    " << x2 << "    " << x3 << "    ";
    file << endl;
    file.close();
}

__host__ void ClearFile(const std::string &file_name) {
    ofstream file(file_name, std::ios::trunc);
    file.close();
    file.clear();
}


