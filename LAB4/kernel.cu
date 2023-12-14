#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int BLOCK_SIZE = 128;
using DataType = double;

const DataType G = 6.67e-11;
const DataType eps = 1e-5;

const std::string bodies_dir = "./output/";

__global__ void
getAcceleration(int num_of_bodies, DataType *dev_mass, DataType *dev_radius, DataType *dev_acceleration) {
    int locIdx = threadIdx.x, globIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int locIdx3 = 3 * locIdx, globIdx3 = 3 * globIdx;

    __shared__ DataType shared_mass[BLOCK_SIZE], shared_radius[3 * BLOCK_SIZE];

    DataType r_x = dev_radius[globIdx3], r_y = dev_radius[globIdx3 + 1], r_z = dev_radius[globIdx3 + 2];
    DataType a_x = 0., a_y = 0., a_z = 0.;
    DataType diff_x, diff_y, diff_z;
    DataType distance, fraction;
    int thread3;
    for (int block_start = 0; block_start < num_of_bodies; block_start += BLOCK_SIZE) {
        shared_mass[locIdx] = dev_mass[locIdx + block_start];
        for (int ax = 0; ax < 3; ++ax)
            shared_radius[locIdx3 + ax] = dev_radius[(block_start + locIdx) * 3 + ax];
        __syncthreads();

        for (int thread = 0; thread < BLOCK_SIZE; ++thread) {
            thread3 = 3 * thread;
            if (block_start + thread < num_of_bodies) {
                diff_x = r_x - shared_radius[thread3];
                diff_y = r_y - shared_radius[thread3 + 1];
                diff_z = r_z - shared_radius[thread3 + 2];

                distance = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
                distance *= __fsqrt_rd(distance);
                fraction = __fdividef(shared_mass[thread], fmaxf(distance, eps));

                a_x += diff_x * fraction;
                a_y += diff_y * fraction;
                a_z += diff_z * fraction;
            }
            __syncthreads();
        }
    }

    if (globIdx < num_of_bodies) {
        dev_acceleration[globIdx3] = -G * a_x;
        dev_acceleration[globIdx3 + 1] = -G * a_y;
        dev_acceleration[globIdx3 + 2] = -G * a_z;
    }
}

__global__ void
calcRungeKuttaStep(int num_of_bodies, DataType *next_y, DataType *y, DataType *deriv_y, DataType time_step) {
    int globIdx = threadIdx.x + blockDim.x * blockIdx.x;
    int globIdx3 = 3 * globIdx;
    if (globIdx < num_of_bodies) {
        for (int ax = 0; ax < 3; ax++) {
            next_y[globIdx3 + ax] = y[globIdx3 + ax] + time_step * deriv_y[3 * globIdx + ax];
        }
    }
}

void printBodyPosition(DataType time, int body, const std::vector <DataType> &point) {
    int body3 = 3 * body;
    std::cout << time << ' ' << point[body3 + 0] << ' ' << point[body3 + 1] << ' ' << point[body3 + 2] << std::endl;
}

void solveNBodyRungeKutta2(DataType time_step, DataType end_time, int num_of_bodies, \
    std::vector <DataType> &weight, std::vector <DataType> &point, std::vector <DataType> &velocity, \
    int save_frequency = 10, bool do_time_output = 1, bool do_file_output = 0) {
    int num_of_bodies3 = 3 * num_of_bodies;

    DataType *dev_mass, *dev_radius, *dev_velocity, *dev_acceleration;
    DataType *dev_radius_buffer, *dev_velocity_buffer;

    cudaMalloc(&dev_mass, num_of_bodies * sizeof(DataType));
    cudaMalloc(&dev_radius, num_of_bodies3 * sizeof(DataType));
    cudaMalloc(&dev_velocity, num_of_bodies3 * sizeof(DataType));
    cudaMalloc(&dev_acceleration, num_of_bodies3 * sizeof(DataType));
    cudaMalloc(&dev_radius_buffer, num_of_bodies3 * sizeof(DataType));
    cudaMalloc(&dev_velocity_buffer, num_of_bodies3 * sizeof(DataType));

    cudaMemcpy(dev_mass, weight.data(), num_of_bodies * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_radius, point.data(), num_of_bodies3 * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_velocity, velocity.data(), num_of_bodies3 * sizeof(DataType), cudaMemcpyHostToDevice);

    int num_of_blocks = (num_of_bodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int threads_per_block = BLOCK_SIZE;

    std::vector <std::ofstream> files(num_of_bodies);
    if (do_file_output) {
        int body3;
        for (int body = 0; body < num_of_bodies; ++body) {
            body3 = 3 * body;
            files[body].open(bodies_dir + std::to_string(num_of_bodies) + "_Body_" + std::to_string(body + 1) + ".txt");
            files[body] << 0. << ' ' << point[body3 + 0] << ' ' << point[body3 + 1] << ' ' << point[body3 + 2]
                        << std::endl;
        }
    }

    cudaEvent_t start, finish, prev_iter, iter;
    cudaEventCreate(&finish);
    cudaEventCreate(&iter);
    cudaEventCreate(&start);
    cudaEventCreate(&prev_iter);
    cudaEventRecord(start);
    cudaEventRecord(prev_iter);
    cudaEventSynchronize(start);
    cudaEventSynchronize(prev_iter);
    DataType iter_time;
    int last_step = std::ceil(end_time / time_step);
    for (int step = 1; step < last_step + 1; ++step) {
        getAcceleration<<<num_of_blocks, threads_per_block>>>(num_of_bodies, dev_mass, dev_radius, dev_acceleration);
        calcRungeKuttaStep<<<num_of_blocks, threads_per_block>>>(num_of_bodies, dev_radius_buffer, dev_radius,
                                                                 dev_velocity, 0.5 * time_step);
        calcRungeKuttaStep<<<num_of_blocks, threads_per_block>>>(num_of_bodies, dev_velocity_buffer, dev_velocity,
                                                                 dev_acceleration, 0.5 * time_step);

        getAcceleration<<<num_of_blocks, threads_per_block>>>(num_of_bodies, dev_mass, dev_radius_buffer,
                                                              dev_acceleration);
        calcRungeKuttaStep<<<num_of_blocks, threads_per_block>>>(num_of_bodies, dev_radius, dev_radius,
                                                                 dev_velocity_buffer, time_step);
        calcRungeKuttaStep<<<num_of_blocks, threads_per_block>>>(num_of_bodies, dev_velocity, dev_velocity,
                                                                 dev_acceleration, time_step);

        if (step % save_frequency == 0) {
            if (do_time_output) {
                cudaEventRecord(iter);
                cudaEventSynchronize(iter);
                cudaEventElapsedTime(&iter_time, prev_iter, iter);
                printf("Iter time = %f\n", iter_time / (save_frequency * 1000.));
                cudaEventRecord(prev_iter);
                cudaEventSynchronize(prev_iter);
            }
            if (do_file_output) {
                if (cudaSuccess !=
                    cudaMemcpy(point.data(), dev_radius, num_of_bodies3 * sizeof(DataType), cudaMemcpyDeviceToHost))
                    printf("Error in cudaMemcpy from Dev to Host!\n");
                DataType curr_time = step * time_step;
                int body3;
                for (int body = 0; body < num_of_bodies; ++body) {
                    body3 = 3 * body;
                    files[body] << curr_time << ' ' << point[body3 + 0] << ' ' << point[body3 + 1] << ' '
                                << point[body3 + 2] << std::endl;
                }
            }
        }
    }
    cudaEventRecord(finish);
    cudaEventSynchronize(finish);

    DataType calc_time;
    cudaEventElapsedTime(&calc_time, start, finish);
    printf("Total time = %f\n", calc_time / 1000.);
    cudaEventDestroy(start);
    cudaEventDestroy(finish);
    cudaEventDestroy(prev_iter);
    cudaEventDestroy(iter);

    if (do_file_output)
        for (size_t body = 0; body < num_of_bodies; ++body)
            files[body].close();

    cudaFree(dev_mass);
    cudaFree(dev_radius);
    cudaFree(dev_velocity);
    cudaFree(dev_acceleration);
    cudaFree(dev_radius_buffer);
    cudaFree(dev_velocity_buffer);
}

int getBodyData(std::vector <DataType> &weight, std::vector <DataType> &point, std::vector <DataType> &velocity, \
     const std::string &file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cout << "Could not open the file!" << std::endl;
        return 0;
    }
    int num_of_bodies;
    file >> num_of_bodies;
    weight.resize(num_of_bodies);
    point.resize(3 * num_of_bodies);
    velocity.resize(3 * num_of_bodies);
    int body3;
    for (int body = 0; body < num_of_bodies; ++body) {
        body3 = body * 3;
        file >> weight[body]
             >> point[body3] >> point[body3 + 1] >> point[body3 + 2]
             >> velocity[body3] >> velocity[body3 + 1] >> velocity[body3 + 2];
    }
    file.close();
    return num_of_bodies;
}

int main(int argc, char **argv) {
    std::string file_path = "Input.txt";
    std::vector <DataType> weight, point, velocity;
    int num_of_bodies = getBodyData(weight, point, velocity, file_path);
    if (!num_of_bodies)
        return 1;

    DataType time_step = 0.01;
    DataType end_time = 5.;
    int save_frequency = 10;
    bool do_time_output = 1;
    bool do_file_output = 0;
    solveNBodyRungeKutta2(time_step, end_time, num_of_bodies, weight, point, velocity, save_frequency, do_time_output,
                          do_file_output);
    return 0;
}