#include <iostream>
#include <mpi.h>
#include <cmath>
#include <vector>
#include "helmholtz.h"

void elementsDistributionCalc(std::vector<int> &elementNumber, std::vector<int> &displacementOfElement, const int np) {
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
    for (int i = 1; i < np; ++i) {
        displacementOfElement[i] = displacementOfElement[i - 1] + elementNumber[i - 1];
    }

    for (int i = 0; i < np; ++i) {
        elementNumber[i] += 2 * N;
    }
    elementNumber[0] -= N;
    elementNumber[np - 1] -= N;
}

int main(int argc, char **argv) {
    int myId, np, iterationsCount;
    double t1, t2, t3, t4, norm_f;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);
    
    std::vector<int> elementNumber(np);
    std::vector<int> displacementOfElement(np);

    if (myId == 0) {
        elementsDistributionCalc(elementNumber, displacementOfElement, np);
    }

    MPI_Bcast(elementNumber.data(), np, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displacementOfElement.data(), np, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> solution;
    std::vector<double> tempSolution;
    std::vector<double> resultSolve;
    std::vector<double> preciseVectorSolution;
    
    if (myId == 0) {
        std::cout << "np: " << np << std::endl << std::endl;
        resultSolve.resize(N * N, 0);
        preciseVectorSolution.resize(N * N);
        Helmholtz::preciseSolution(preciseVectorSolution);
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
        norm_f = Helmholtz::jacobiMethod(solution, tempSolution, elementNumber, myId, np, iterationsCount, JacobiNone);
        t2 = MPI_Wtime();
        std::cout << std::endl << "jacobiMethod seq: " << std::endl;
        std::cout << "Time = " << t2 - t1 << std::endl;
        std::cout << "Iterations = " << iterationsCount << std::endl;
        std::cout << "Error = " << norm_f << std::endl;
        std::cout << "|solution - preciseVectorSolution| = "
                  << Helmholtz::norm(solution, preciseVectorSolution, 0, N * N) << std::endl << std::endl;

        std::fill(solution.begin(), solution.end(), 0.0);
        std::fill(tempSolution.begin(), tempSolution.end(), 0.0);

        t3 = MPI_Wtime();
        norm_f = Helmholtz::redAndBlackMethod(solution, tempSolution, elementNumber, myId, np, iterationsCount,
                                              RedAndBlackNone);
        t4 = MPI_Wtime();
        std::cout << std::endl << "redAndBlackM seq: " << std::endl;
        std::cout << "Time = " << t4 - t3 << std::endl;
        std::cout << "Iterations = " << iterationsCount << std::endl;
        std::cout << "Error = " << norm_f << std::endl;
        std::cout << "|solution - preciseVectorSolution| = "
                  << Helmholtz::norm(solution, preciseVectorSolution, 0, N * N) << std::endl << std::endl;
    }
    for (int methodType = 0; methodType < 3 && np > 1; ++methodType) {
        solution.resize(elementNumber[myId], 0);
        std::fill(solution.begin(), solution.end(), 0.0);
        tempSolution.resize(elementNumber[myId], 0);
        std::fill(tempSolution.begin(), tempSolution.end(), 0.0);

        t1 = MPI_Wtime();
        norm_f = Helmholtz::jacobiMethod(solution, tempSolution, elementNumber, myId, np, iterationsCount,
                                         static_cast<JacobiSolutionMethod>(methodType));
        t2 = MPI_Wtime();
        if (myId == 0) {
            std::cout << "exec time: " << t2 - t1 << std::endl;
            std::cout << "iterationsCount: " << iterationsCount << std::endl;
            std::cout << "error: " << norm_f << std::endl;
        }
        Helmholtz::gatherSolution(elementNumber, solution, resultSolve, displacementOfElement, np, myId);
        if (myId == 0) {
            std::cout << "diff with precise solution: " << Helmholtz::norm(resultSolve, preciseVectorSolution, 0, N * N)
                      << std::endl
                      << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        solution.resize(elementNumber[myId], 0);
        std::fill(solution.begin(), solution.end(), 0.0);
        tempSolution.resize(elementNumber[myId], 0);
        std::fill(tempSolution.begin(), tempSolution.end(), 0.0);

        t1 = MPI_Wtime();
        norm_f = Helmholtz::redAndBlackMethod(solution, tempSolution, elementNumber, myId, np, iterationsCount,
                                              static_cast<RedAndBlackSolutionMethod>(methodType));
        t2 = MPI_Wtime();
        if (myId == 0) {
            std::cout << "exec time: " << t2 - t1 << std::endl;
            std::cout << "iterationsCount: " << iterationsCount << std::endl;
            std::cout << "error: " << norm_f << std::endl;
        }
        Helmholtz::gatherSolution(elementNumber, solution, resultSolve, displacementOfElement, np, myId);
        if (myId == 0) {
            std::cout << "diff with precise solution: " << Helmholtz::norm(resultSolve, preciseVectorSolution, 0, N * N)
                      << std::endl
                      << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Finalize();
}
