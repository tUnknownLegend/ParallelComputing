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

enum AllMethodTypes {
    redBlackNone = -2,
    jacobiNone = -1,
    jacobiSendRecv,
    jacobiSendAndRecv,
    jacobiISendIRecv,
    redBlackSendRecv,
    redBlackSendAndRecv,
    redBlackISendIRecv,
};

void
printAndCalcResults(std::vector<double> &solution, std::vector<double> &tempSolution,
                    std::vector<double> &resultSolve,
                    std::vector<double> &preciseVectorSolution,
                    std::vector<int> &elementNumber, std::vector<int> &displacementOfElement,
                    const int myId, const int np, int &iterationsCount, const AllMethodTypes method) {
    solution.resize(elementNumber[myId], 0);
    std::fill(solution.begin(), solution.end(), 0.0);
    tempSolution.resize(elementNumber[myId], 0);
    std::fill(tempSolution.begin(), tempSolution.end(), 0.0);

    const auto startTime = MPI_Wtime();

    switch (method) {
        case redBlackNone:
            Helmholtz::redAndBlackMethod(solution, tempSolution, elementNumber, myId, np, iterationsCount,
                                         RedAndBlackNone);
            break;
        case jacobiNone:
            Helmholtz::jacobiMethod(solution, tempSolution, elementNumber, myId, np, iterationsCount,
                                    JacobiNone);
            break;
        case jacobiSendRecv:
            Helmholtz::jacobiMethod(solution, tempSolution, elementNumber, myId, np, iterationsCount,
                                    JacobiSendReceive);
            break;
        case jacobiSendAndRecv:
            Helmholtz::jacobiMethod(solution, tempSolution, elementNumber, myId, np, iterationsCount,
                                    JacobiSendAndReceive);
            break;
        case jacobiISendIRecv:
            Helmholtz::jacobiMethod(solution, tempSolution, elementNumber, myId, np, iterationsCount,
                                    JacobiISendIReceive);
            break;
        case redBlackSendRecv:
            Helmholtz::redAndBlackMethod(solution, tempSolution, elementNumber, myId, np, iterationsCount,
                                         RedAndBlackSendReceive);
            break;
        case redBlackSendAndRecv:
            Helmholtz::redAndBlackMethod(solution, tempSolution, elementNumber, myId, np, iterationsCount,
                                         RedAndBlackSendAndReceive);
            break;
        case redBlackISendIRecv:
            Helmholtz::redAndBlackMethod(solution, tempSolution, elementNumber, myId, np, iterationsCount,
                                         RedAndBlackISendIReceive);
            break;
        default:
            std::cerr << "method not implemented\n";
    }


    const auto endTime = MPI_Wtime();
    if (myId == 0) {
        std::cout << "exec time: " << endTime - startTime << std::endl;
        std::cout << "iterationsCount: " << iterationsCount << std::endl;
    }
    Helmholtz::gatherSolution(elementNumber, solution, resultSolve, displacementOfElement, np, myId);
    if (myId == 0) {
        std::cout << "diff with precise solution: " << Helmholtz::norm(resultSolve, preciseVectorSolution, 0, N * N)
                  << std::endl
                  << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char **argv) {
    int myId, np, iterationsCount;
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
        std::cout << "number of processors: " << np << std::endl << std::endl;
        resultSolve.resize(N * N, 0);
        preciseVectorSolution.resize(N * N);
        Helmholtz::preciseSolution(preciseVectorSolution);
    }

    if (np == 1) {
        printAndCalcResults(solution, tempSolution, resultSolve, preciseVectorSolution, elementNumber,
                            displacementOfElement, myId, np, iterationsCount, jacobiNone);

        printAndCalcResults(solution, tempSolution, resultSolve, preciseVectorSolution, elementNumber,
                            displacementOfElement, myId, np, iterationsCount, redBlackNone);

    } else {
        for (int methodType = 0; methodType < 6 && np > 1; ++methodType) {
            printAndCalcResults(solution, tempSolution, resultSolve, preciseVectorSolution, elementNumber,
                                displacementOfElement, myId, np, iterationsCount,
                                static_cast<const AllMethodTypes>(methodType));
        }
    }
    MPI_Finalize();
}
