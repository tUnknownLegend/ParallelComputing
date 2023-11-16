#include <iostream>
#include <mpi.h>
#include <cmath>
#include <vector>
#include "helmholtz.h"

int main(int argc, char **argv) {
    int myId, np, iterations;
    double t1, t2, t3, t4, norm_f;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);

    std::vector<double> solution, tempSolution, y_gen, u;
    std::vector<int> elementNumber(np), displacementOfElement(np);

    if (myId == 0) {
        if (N % np == 0) {
            for (int i = 0; i < np; ++i)
                elementNumber[i] = (N / np) * N;
        } else {
            int temp = 0;
            for (int i = 0; i < np - 1; ++i) {
                elementNumber[i] = round(((double) N / (double) np)) * N;
                temp += elementNumber[i] / N;
            }
            elementNumber[np - 1] = (N - temp) * N;
        }

        displacementOfElement[0] = 0;
        for (int i = 1; i < np; ++i)
            displacementOfElement[i] = displacementOfElement[i - 1] + elementNumber[i - 1];

        for (int i = 0; i < np; ++i)
            elementNumber[i] += 2 * N;
        elementNumber[0] -= N;
        elementNumber[np - 1] -= N;
    }

    MPI_Bcast(elementNumber.data(), np, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displacementOfElement.data(), np, MPI_INT, 0, MPI_COMM_WORLD);


    if (myId == 0) {
        std::cout << "np: " << np << std::endl << std::endl;
        y_gen.resize(N * N, 0);
        u.resize(N * N);
        Helmholtz::preciseSolution(u);
    }

    if (np == 1) {
        solution.resize(elementNumber[myId], 0);
        std::fill(solution.begin(), solution.end(), 0.0);
        tempSolution.resize(elementNumber[myId], 0);
        std::fill(tempSolution.begin(), tempSolution.end(), 0.0);

        solution.resize(N * N, 0);
        std::fill(solution.begin(), solution.end(), 0.0);
        tempSolution.resize(N * N, 0);
        std::fill(tempSolution.begin(), tempSolution.end(), 0.0);


        t1 = MPI_Wtime();
        norm_f = Helmholtz::jacobiMethod(solution, tempSolution, elementNumber, myId, np, iterations, JacobiNone);
        t2 = MPI_Wtime();
        std::cout << std::endl << "jacobiMethod seq: " << std::endl;
        std::cout << "Time = " << t2 - t1 << std::endl;
        std::cout << "Iterations = " << iterations << std::endl;
        std::cout << "Error = " << norm_f << std::endl;
        std::cout << "|solution - u| = " << Helmholtz::norm(solution, u, 0, N * N) << std::endl << std::endl;

        std::fill(solution.begin(), solution.end(), 0.0);
        std::fill(tempSolution.begin(), tempSolution.end(), 0.0);

        t3 = MPI_Wtime();
        norm_f = Helmholtz::redAndBlackMethod(solution, tempSolution, elementNumber, myId, np, iterations,
                                              RedAndBlackNone);
        t4 = MPI_Wtime();
        std::cout << std::endl << "redAndBlackM seq: " << std::endl;
        std::cout << "Time = " << t4 - t3 << std::endl;
        std::cout << "Iterations = " << iterations << std::endl;
        std::cout << "Error = " << norm_f << std::endl;
        std::cout << "|solution - u| = " << Helmholtz::norm(solution, u, 0, N * N) << std::endl << std::endl;
    }
    for (int methodType = 0; methodType < 3 && np > 1; ++methodType) {
        solution.resize(elementNumber[myId], 0);
        std::fill(solution.begin(), solution.end(), 0.0);
        tempSolution.resize(elementNumber[myId], 0);
        std::fill(tempSolution.begin(), tempSolution.end(), 0.0);

        t1 = MPI_Wtime();
        norm_f = Helmholtz::jacobiMethod(solution, tempSolution, elementNumber, myId, np, iterations,
                                         static_cast<JacobiSolutionMethod>(methodType));
        t2 = MPI_Wtime();
        if (myId == 0) {
            std::cout << "exec time: " << t2 - t1 << std::endl;
            std::cout << "iterations: " << iterations << std::endl;
            std::cout << "error: " << norm_f << std::endl;
        }
        Helmholtz::gatherSolution(elementNumber, solution, y_gen, displacementOfElement, np, myId);
        if (myId == 0)
            std::cout << "diff with precise solution: " << Helmholtz::norm(y_gen, u, 0, N * N) << std::endl
                      << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);

        solution.resize(elementNumber[myId], 0);
        std::fill(solution.begin(), solution.end(), 0.0);
        tempSolution.resize(elementNumber[myId], 0);
        std::fill(tempSolution.begin(), tempSolution.end(), 0.0);

        t1 = MPI_Wtime();
        norm_f = Helmholtz::redAndBlackMethod(solution, tempSolution, elementNumber, myId, np, iterations,
                                              static_cast<RedAndBlackSolutionMethod>(methodType));
        t2 = MPI_Wtime();
        if (myId == 0) {
            std::cout << "exec time: " << t2 - t1 << std::endl;
            std::cout << "iterations: " << iterations << std::endl;
            std::cout << "error: " << norm_f << std::endl;
        }
        Helmholtz::gatherSolution(elementNumber, solution, y_gen, displacementOfElement, np, myId);
        if (myId == 0)
            std::cout << "diff with precise solution: " << Helmholtz::norm(y_gen, u, 0, N * N) << std::endl
                      << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Finalize();
}
