#include <iostream>
#include <mpi.h>
#include <cmath>
#include <vector>
#include "helmholtz.h"

int main(int argc, char **argv) {
    int myid, np, iterations;
    double t1, t2, t3, t4, norm_f;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    std::vector<double> y, y_n, y_gen, u;
    std::vector<int> el_num(np), displs(np);

    if (myid == 0) {
        if (N % np == 0) {
            for (int i = 0; i < np; ++i)
                el_num[i] = (N / np) * N;
        } else {
            int temp = 0;
            for (int i = 0; i < np - 1; ++i) {
                el_num[i] = round(((double) N / (double) np)) * N;
                temp += el_num[i] / N;
            }
            el_num[np - 1] = (N - temp) * N;
        }

        displs[0] = 0;
        for (int i = 1; i < np; ++i)
            displs[i] = displs[i - 1] + el_num[i - 1];

        for (int i = 0; i < np; ++i)
            el_num[i] += 2 * N;
        el_num[0] -= N;
        el_num[np - 1] -= N;
    }

    MPI_Bcast(el_num.data(), np, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs.data(), np, MPI_INT, 0, MPI_COMM_WORLD);


    if (myid == 0) {
        std::cout << "np: " << np << std::endl << std::endl;
        y_gen.resize(N * N, 0);
        u.resize(N * N);
        Helmholtz::analyt_sol(u);
    }

    if (np == 1) {
        y.resize(el_num[myid], 0);
        Helmholtz::zero(y);
        y_n.resize(el_num[myid], 0);
        Helmholtz::zero(y_n);

        y.resize(N * N, 0);
        Helmholtz::zero(y);
        y_n.resize(N * N, 0);
        Helmholtz::zero(y_n);


        t1 = MPI_Wtime();
        norm_f = Helmholtz::Jacobi(y, y_n, el_num, myid, np, iterations, 0);
        t2 = MPI_Wtime();
        std::cout << std::endl << "Jacobi seq" << std::endl;
        std::cout << "Time = " << t2 - t1 << std::endl;
        std::cout << "Iterations = " << iterations << std::endl;
        std::cout << "Error = " << norm_f << std::endl;
        std::cout << "|y - u| = " << Helmholtz::norm(y, u, 0, N * N) << std::endl << std::endl;

        Helmholtz::zero(y);
        Helmholtz::zero(y_n);

        t3 = MPI_Wtime();
        norm_f = Helmholtz::Zeidel(y, y_n, el_num, myid, np, iterations, 0);
        t4 = MPI_Wtime();
        std::cout << std::endl << "Zeidel seq" << std::endl;
        std::cout << "Time = " << t4 - t3 << std::endl;
        std::cout << "Iterations = " << iterations << std::endl;
        std::cout << "Error = " << norm_f << std::endl;
        std::cout << "|y - u| = " << Helmholtz::norm(y, u, 0, N * N) << std::endl << std::endl;
    }
    for (int send_type = 1; send_type <= 3; ++send_type) {
        if (np > 1) {
            y.resize(el_num[myid], 0);
            Helmholtz::zero(y);
            y_n.resize(el_num[myid], 0);
            Helmholtz::zero(y_n);

            t1 = MPI_Wtime();
            norm_f = Helmholtz::Jacobi(y, y_n, el_num, myid, np, iterations, send_type);
            t2 = MPI_Wtime();
            if (myid == 0) {
                std::cout << "Time = " << t2 - t1 << std::endl;
                std::cout << "Iterations = " << iterations << std::endl;
                std::cout << "Error = " << norm_f << std::endl;
            }
            Helmholtz::generalY(el_num, y, y_gen, displs, np, myid);
            if (myid == 0)
                std::cout << "|y - u| = " << Helmholtz::norm(y_gen, u, 0, N * N) << std::endl << std::endl;
            MPI_Barrier(MPI_COMM_WORLD);

            y.resize(el_num[myid], 0);
            Helmholtz::zero(y);
            y_n.resize(el_num[myid], 0);
            Helmholtz::zero(y_n);

            t1 = MPI_Wtime();
            norm_f = Helmholtz::Zeidel(y, y_n, el_num, myid, np, iterations, send_type);
            t2 = MPI_Wtime();
            if (myid == 0) {
                std::cout << "Time = " << t2 - t1 << std::endl;
                std::cout << "Iterations = " << iterations << std::endl;
                std::cout << "Error = " << norm_f << std::endl;
            }
            Helmholtz::generalY(el_num, y, y_gen, displs, np, myid);
            if (myid == 0)
                std::cout << "|y - u| = " << Helmholtz::norm(y_gen, u, 0, N * N) << std::endl << std::endl;
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
}
