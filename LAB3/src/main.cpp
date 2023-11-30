#include <iostream>
#include <cmath>
#include <mpi.h>
#include <fstream>
#include "nBodyTask.h"
#include "shared.h"

using std::pair;
using std::ostream;
using std::endl;
using std::cout;


int main(int argc, char **argv) {
    // != 0 - считывать из файла, 0 - заполнять случайно
    int doReadFromFile = 1;
    // != 0 - записывать в файлы, 0 - нет
    int doWriteToFile = 1;

    // Номер текущего процесса
    int myId = 0;
    // Общее число всех процессов
    int np = 0;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &myId);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    Body *data = nullptr; // Массив "тел"

    // Массив длин
    int *locSize = new int[np];
    // массив смещений локальных массивов
    int *locOffset = new int[np];

    if (myId == 0) {
        if (doReadFromFile) {
            data = new Body[N];
            read_file("4body.txt", data, N);
        } else {
            data = new Body[N];

            for (int i = 0; i < N; ++i) {
                data[i].weight = GetRandomDouble(weightRange.first, weightRange.second);

                for (int k = 0; k < 3; ++k) {
                    data[i].position[k] = GetRandomDouble(positionRange.first, positionRange.second);
                    data[i].velocity[k] = GetRandomDouble(velocityRange.first, velocityRange.second);
                }
            }
        }

        locOffset[0] = 0;

        // Размер частей массива
        int L = N / np;

        for (int p = 0; p < np - 1; ++p) {
            locSize[p] = L;
            locOffset[p + 1] = locOffset[p] + L;
        }

        locSize[np - 1] = L + N % np;
    }

    //

    MPI_Bcast(locSize, np, MPI_INT, 0, MPI_COMM_WORLD);   // Рассылка всем
    MPI_Bcast(locOffset, np, MPI_INT, 0, MPI_COMM_WORLD); // процессам

    Body *locDat = new Body[locSize[myId]]; // Локальный массив "тел"

    int blockLengths[3] = {1, 3, 3};

    MPI_Datatype types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};

    MPI_Aint offsets[3] = {offsetof(Body, weight), offsetof(Body, position), offsetof(Body, velocity)};

    MPI_Datatype MPI_Body; // Новый тип данных - структура "тело"

    MPI_Type_create_struct(3, blockLengths, offsets, types, &MPI_Body);

    MPI_Type_commit(&MPI_Body);

    // Тип только для позиций
    int lengthsR[] = {3};
    MPI_Aint offsetsR[] = {offsetof(Body, position)};
    MPI_Datatype typesR[] = {MPI_DOUBLE};

    MPI_Datatype MPI_Helptype; // Вспомогательный тип
    MPI_Type_create_struct(1, lengthsR, offsetsR, typesR, &MPI_Helptype);

    MPI_Datatype mpi_Body_r;
    MPI_Type_create_resized(MPI_Helptype, 0, 56, &mpi_Body_r); // Body состоит из 7 double => 7 x 8 = 56

    MPI_Type_commit(&mpi_Body_r);


    // Рассылка всем процессам частей массива "тел"
    MPI_Scatterv(data, locSize, locOffset, MPI_Body, locDat, locSize[myId], MPI_Body, 0, MPI_COMM_WORLD);

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD); // Рассылка числа тел всем процессам

    if (myId) {
        data = new Body[N]; // Создание массива "тел" для остальных процессов
    }

    MPI_Bcast(data, N, MPI_Body, 0, MPI_COMM_WORLD); // Рассылка массива тел всем процессам

    MPI_Bcast(&doWriteToFile, 1, MPI_INT, 0, MPI_COMM_WORLD); // Рассылка флага записи в файлы всем процессам

    std::ofstream *F = nullptr;

    if (doWriteToFile) {
        F = new std::ofstream[locSize[myId]]; // Массив файлов для каждого узла

        for (int i = 0; i < locSize[myId]; ++i) {
            F[i].open(std::to_string(num) + "_Body_" + std::to_string(locOffset[myId] + i + 1) + ".txt");

            //F[i].open("Body_" + std::to_string(locOffset[myId] + i + 1) + ".txt");

            //F[i] << N << " " << locDat[i].weight << " " << tau << std::endl;

            F[i] << 0.0 << " " << locDat[i];
        }
    }

    Body bod_i{}; // Текущее тело


    double a[3] = {0.0, 0.0, 0.0}; // Текущие ускорения
    Body *locDatBuf = new Body[locSize[myId]];  // Промежуточный локальный массив "тел"
    auto *w = new double[3 * locSize[myId]];  // Начальные ускорения

    const double start = MPI_Wtime();

    // Расчётная схема
    for (int t = 1; t <= Nt; ++t) {
        for (int i = 0; i < locSize[myId]; ++i) {
            bod_i = locDat[i];

            locDatBuf[i] = bod_i;

            acceleration(a, N, bod_i.position, data, G);

            for (int k = 0; k < 3; ++k) {
                locDatBuf[i].position[k] += tau * bod_i.velocity[k];

                locDatBuf[i].velocity[k] += tau * a[k];

                w[3 * i + k] = a[k];
            }
        }

        MPI_Allgatherv(locDatBuf, locSize[myId], mpi_Body_r, data, locSize, locOffset, mpi_Body_r, MPI_COMM_WORLD);

        for (int i = 0; i < locSize[myId]; ++i) {

            acceleration(a, N, locDatBuf[i].position, data, G);

            for (int k = 0; k < 3; ++k) {
                locDat[i].position[k] += 0.5 * tau * (locDat[i].velocity[k] + locDatBuf[i].velocity[k]);
                locDat[i].velocity[k] += 0.5 * tau * (w[3 * i + k] + a[k]);
            }

            if (t % tf == 0 && doWriteToFile)
                F[i] << t * tau << " " << locDat[i];
        }

        MPI_Allgatherv(locDat, locSize[myId], MPI_Body, data, locSize, locOffset, MPI_Body, MPI_COMM_WORLD);
    }

    const double end = MPI_Wtime();

    //

    if (myId == 0) {
        cout << "time = " << end - start << endl;
    }


    if (doWriteToFile)
        for (int i = 0; i < locSize[myId]; ++i)
            F[i].close();

    delete[] F;
    delete[] data;
    delete[] locDat;
    delete[] locSize;
    delete[] locOffset;

    MPI_Finalize();

    return 0;
}