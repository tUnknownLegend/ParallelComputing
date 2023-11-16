#include "helmholtz.h"
#include "shared.h"
#include <mpi.h>
#include <iostream>

using std::vector;
using std::pair;
using std::pow;
using std::cerr;
using std::cout;
using std::function;
using std::abs;
using std::max;

double
Helmholtz::jacobiMethod(vector<double> &solution, vector<double> &tempSolution, vector<int> &elementNumber, int myId,
                        int np,
                        int &iterationCount,
                        const JacobiSolutionMethod methodType) {
    double normValue = 0;
    if (np == 1) {

        iterationCount = 0;
        do {
            ++iterationCount;
            for (int i = 1; i < N - 1; ++i) {
                for (int j = 1; j < N - 1; ++j) {
                    solution[i * N + j] = (h * h * rightSideFunction(i * h, j * h) +
                                           (tempSolution[i * N + j - 1] + tempSolution[i * N + j + 1] +
                                            tempSolution[(i - 1) * N + j] +
                                            tempSolution[(i + 1) * N + j])) / multiplayer;
                }
            }
            normValue = norm(solution, tempSolution, 0, N * N);
            tempSolution.swap(solution);
        } while (normValue > COMPARE_RATE);
    }
    if (np > 1) {
        switch (methodType) {
            case JacobiSendReceive:
                normValue = solveMPI(solution, tempSolution, elementNumber, myId, np, iterationCount, JacobiSendRecv);
                break;
            case JacobiSendAndReceive:
                normValue = solveMPI(solution, tempSolution, elementNumber, myId, np, iterationCount,
                                     JacobiSendAndRecv);
                break;
            case JacobiISendIReceive:
                normValue = solveMPI(solution, tempSolution, elementNumber, myId, np, iterationCount, JacobiISendIRecv);
                break;
            default:
                cerr << methodType << ". method not implemented\n";
        }
    }
    if (myId == 0) {
        switch (methodType) {
            case JacobiSendReceive:
                cout << methodType << ". JacobiSendRecv\n";
                break;
            case JacobiSendAndReceive:
                cout << methodType << ". JacobiSendAndRecv\n";
                break;
            case JacobiISendIReceive:
                cout << methodType << ". JacobiISendIRecv\n";
                break;
            default:
                cerr << methodType << ". method not implemented\n";
        }
    }
    return normValue;
}

double
Helmholtz::redAndBlackMethod(vector<double> &solution, vector<double> &tempSolution, vector<int> elementNumber,
                             const int myId,
                             const int np,
                             int &iterationCount,
                             const RedAndBlackSolutionMethod methodType) {
    double normValue;
    if (np == 1) {
        iterationCount = 0;
        do {
            ++iterationCount;
            for (int i = 1; i < N - 1; ++i) {
                for (int j = (i % 2) + 1; j < N - 1; j += 2) {
                    solution[i * N + j] = (h * h * rightSideFunction(i * h, j * h) +
                                           (tempSolution[i * N + j - 1] + tempSolution[i * N + j + 1] +
                                            tempSolution[(i - 1) * N + j] +
                                            tempSolution[(i + 1) * N + j])) / multiplayer;
                }
            }

            for (int i = 1; i < N - 1; ++i) {
                for (int j = ((i + 1) % 2) + 1; j < N - 1; j += 2) {
                    solution[i * N + j] = (h * h * rightSideFunction(i * h, j * h) +
                                           (solution[i * N + j - 1] + solution[i * N + j + 1] +
                                            solution[(i - 1) * N + j] + solution[(i + 1) * N + j])) /
                                          multiplayer;
                }
            }
            normValue = norm(solution, tempSolution, 0, N * N);
            tempSolution.swap(solution);
        } while (normValue > COMPARE_RATE);
    }
    if (np > 1) {
        switch (methodType) {
            case RedAndBlackSendReceive:
                normValue = solveMPI(solution, tempSolution, elementNumber, myId, np, iterationCount,
                                     redAndBlackSendRecv);
                break;
            case RedAndBlackSendAndReceive:
                normValue = solveMPI(solution, tempSolution, elementNumber, myId, np, iterationCount,
                                     redAndBlackSendAndRecv);
                break;
            case RedAndBlackISendIReceive:
                normValue = solveMPI(solution, tempSolution, elementNumber, myId, np, iterationCount,
                                     redAndBlackISendIRecv);
                break;
            default:
                cerr << "method not implemented";
        }
    }
    if (myId == 0) {
        switch (methodType) {
            case RedAndBlackSendReceive:
                cout << methodType << ". redAndBlackMethodSendRecv\n";
                break;
            case RedAndBlackSendAndReceive:
                cout << methodType << ". redAndBlackMethodSendAndRecv\n";
                break;
            case RedAndBlackISendIReceive:
                cout << methodType << ". redAndBlackMethodISendIRecv\n";
                break;
            default:
                cerr << methodType << ". method not implemented\n";
        }
    }
    return normValue;
}

double Helmholtz::solveMPI(vector<double> &solution, vector<double> &tempSolution, vector<int> &elementNumber, int myId,
                           int np, int &iterationCount,
                           const function<void(vector<double> &solution, vector<double> &tempSolution,
                                               vector<int> &elementNumber, int myId,
                                               int np, int &shift)> &calc) {
    double normValue;
    int shift = 0;
    for (int i = 0; i < myId; ++i)
        shift += elementNumber[i] / N;
    shift -= (myId == 0) ? 0 : myId * 2;

    iterationCount = 0;
    double norma;

    do {
        ++iterationCount;

        calc(solution, tempSolution, elementNumber, myId, np, shift);

        norma = norm(solution, tempSolution, (myId == 0) ? 0 : N,
                     (myId == np) ? elementNumber[myId] : elementNumber[myId] - N);
        MPI_Allreduce(&norma, &normValue, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        tempSolution.swap(solution);
    } while (normValue > COMPARE_RATE);

    return normValue;
}

inline void
Helmholtz::JacobiSendRecv(vector<double> &solution, vector<double> &tempSolution, vector<int> &elementNumber, int myId,
                          int np, int &shift) {
    MPI_Send(tempSolution.data() + elementNumber[myId] - 2 * N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
             (myId != np - 1) ? myId + 1 : 0, 1, MPI_COMM_WORLD);
    MPI_Recv(tempSolution.data(), (myId != 0) ? N : 0, MPI_DOUBLE, (myId != 0) ? myId - 1 : np - 1, 1,
             MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

    MPI_Send(tempSolution.data() + N, (myId != 0) ? N : 0, MPI_DOUBLE, (myId != 0) ? myId - 1 : np - 1,
             2,
             MPI_COMM_WORLD);
    MPI_Recv(tempSolution.data() + elementNumber[myId] - N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
             (myId != np - 1) ? myId + 1 : 0, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

    for (int i = 1; i < elementNumber[myId] / N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                   (tempSolution[i * N + j - 1] + tempSolution[i * N + j + 1] +
                                    tempSolution[(i - 1) * N + j] +
                                    tempSolution[(i + 1) * N + j])) / multiplayer;
        }
    }
};

inline void
Helmholtz::JacobiSendAndRecv(vector<double> &solution, vector<double> &tempSolution, vector<int> &elementNumber,
                             int myId,
                             int np, int &shift) {
    MPI_Sendrecv(tempSolution.data() + elementNumber[myId] - 2 * N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                 (myId != np - 1) ? myId + 1 : 0, 3, tempSolution.data(), (myId != 0) ? N : 0,
                 MPI_DOUBLE,
                 (myId != 0) ? myId - 1 : np - 1, 3, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

    MPI_Sendrecv(tempSolution.data() + N, (myId != 0) ? N : 0, MPI_DOUBLE,
                 (myId != 0) ? myId - 1 : np - 1, 4,
                 tempSolution.data() + elementNumber[myId] - N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                 (myId != np - 1) ? myId + 1 : 0, 4, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

    for (int i = 1; i < elementNumber[myId] / N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                   (tempSolution[i * N + j - 1] + tempSolution[i * N + j + 1] +
                                    tempSolution[(i - 1) * N + j] +
                                    tempSolution[(i + 1) * N + j])) / multiplayer;
        }
    }
}

inline void
Helmholtz::JacobiISendIRecv(vector<double> &solution, vector<double> &tempSolution, vector<int> &elementNumber,
                            int myId,
                            int np, int &shift) {
    MPI_Request req_send_up, req_recv_up, req_send_down, req_recv_down;
    if (myId != np - 1) {
        MPI_Isend(tempSolution.data() + elementNumber[myId] - 2 * N, N, MPI_DOUBLE, myId + 1, 5,
                  MPI_COMM_WORLD,
                  &req_send_up);

        MPI_Irecv(tempSolution.data() + elementNumber[myId] - N, N, MPI_DOUBLE, myId + 1, 6, MPI_COMM_WORLD,
                  &req_recv_up);
    }
    if (myId != 0) {
        MPI_Irecv(tempSolution.data(), N, MPI_DOUBLE, myId - 1, 5, MPI_COMM_WORLD, &req_recv_down);

        MPI_Isend(tempSolution.data() + N, N, MPI_DOUBLE, myId - 1, 6, MPI_COMM_WORLD, &req_send_down);
    }

    for (int i = 2; i < elementNumber[myId] / N - 2; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                   (tempSolution[i * N + j - 1] + tempSolution[i * N + j + 1] +
                                    tempSolution[(i - 1) * N + j] +
                                    tempSolution[(i + 1) * N + j])) / multiplayer;
        }
    }
    if (myId != 0) {
        MPI_Wait(&req_recv_down, MPI_STATUSES_IGNORE);
    }
    if (myId != np - 1) {
        MPI_Wait(&req_recv_up, MPI_STATUSES_IGNORE);
    }
    int i = 1;
    for (int j = 1; j < N - 1; ++j) {
        solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                               (tempSolution[i * N + j - 1] + tempSolution[i * N + j + 1] +
                                tempSolution[(i - 1) * N + j] +
                                tempSolution[(i + 1) * N + j])) / multiplayer;
    }

    i = elementNumber[myId] / N - 2;
    for (int j = 1; j < N - 1; ++j) {
        solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                               (tempSolution[i * N + j - 1] + tempSolution[i * N + j + 1] +
                                tempSolution[(i - 1) * N + j] +
                                tempSolution[(i + 1) * N + j])) / multiplayer;
    }
}

void
Helmholtz::redAndBlackSendRecv(vector<double> &solution, vector<double> &tempSolution, vector<int> &elementNumber,
                               const int myId,
                               int np, int &shift) {
    MPI_Send(tempSolution.data() + elementNumber[myId] - 2 * N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
             (myId != np - 1) ? myId + 1 : 0, 1, MPI_COMM_WORLD);
    MPI_Recv(tempSolution.data(), (myId != 0) ? N : 0, MPI_DOUBLE, (myId != 0) ? myId - 1 : np - 1, 1,
             MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

    MPI_Send(tempSolution.data() + N, (myId != 0) ? N : 0, MPI_DOUBLE, (myId != 0) ? myId - 1 : np - 1,
             2,
             MPI_COMM_WORLD);
    MPI_Recv(tempSolution.data() + elementNumber[myId] - N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
             (myId != np - 1) ? myId + 1 : 0, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

    for (int i = 1; i < elementNumber[myId] / N - 1; ++i) {
        for (int j = ((i + shift) % 2) + 1; j < N - 1; j += 2) {
            solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                   (tempSolution[i * N + j - 1] + tempSolution[i * N + j + 1] +
                                    tempSolution[(i - 1) * N + j] +
                                    tempSolution[(i + 1) * N + j])) / multiplayer;
        }
    }

    MPI_Send(solution.data() + elementNumber[myId] - 2 * N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
             (myId != np - 1) ? myId + 1 : 0, 1, MPI_COMM_WORLD);
    MPI_Recv(solution.data(), (myId != 0) ? N : 0, MPI_DOUBLE, (myId != 0) ? myId - 1 : np - 1, 1,
             MPI_COMM_WORLD,
             MPI_STATUSES_IGNORE);

    MPI_Send(solution.data() + N, (myId != 0) ? N : 0, MPI_DOUBLE, (myId != 0) ? myId - 1 : np - 1, 2,
             MPI_COMM_WORLD);
    MPI_Recv(solution.data() + elementNumber[myId] - N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
             (myId != np - 1) ? myId + 1 : 0, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

    for (int i = 1; i < elementNumber[myId] / N - 1; ++i) {
        for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2) {
            solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                   (solution[i * N + j - 1] + solution[i * N + j + 1] +
                                    solution[(i - 1) * N + j] +
                                    solution[(i + 1) * N + j])) / multiplayer;
        }
    }
}

void
Helmholtz::redAndBlackSendAndRecv(vector<double> &solution, vector<double> &tempSolution, vector<int> &elementNumber,
                                  const int myId,
                                  const int np, int &shift) {
    MPI_Sendrecv(tempSolution.data() + elementNumber[myId] - 2 * N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                 (myId != np - 1) ? myId + 1 : 0, 3, tempSolution.data(), (myId != 0) ? N : 0,
                 MPI_DOUBLE,
                 (myId != 0) ? myId - 1 : np - 1, 3, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

    MPI_Sendrecv(tempSolution.data() + N, (myId != 0) ? N : 0, MPI_DOUBLE,
                 (myId != 0) ? myId - 1 : np - 1,
                 4,
                 tempSolution.data() + elementNumber[myId] - N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                 (myId != np - 1) ? myId + 1 : 0, 4, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

    for (int i = 1; i < elementNumber[myId] / N - 1; ++i) {
        for (int j = ((i + shift) % 2) + 1; j < N - 1; j += 2) {
            solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                   (tempSolution[i * N + j - 1] + tempSolution[i * N + j + 1] +
                                    tempSolution[(i - 1) * N + j] +
                                    tempSolution[(i + 1) * N + j])) / multiplayer;
        }
    }

    MPI_Sendrecv(solution.data() + elementNumber[myId] - 2 * N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                 (myId != np - 1) ? myId + 1 : 0, 3, solution.data(), (myId != 0) ? N : 0, MPI_DOUBLE,
                 (myId != 0) ? myId - 1 : np - 1, 3, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

    MPI_Sendrecv(solution.data() + N, (myId != 0) ? N : 0, MPI_DOUBLE, (myId != 0) ? myId - 1 : np - 1,
                 4,
                 solution.data() + elementNumber[myId] - N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                 (myId != np - 1) ? myId + 1 : 0, 4, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

    for (int i = 1; i < elementNumber[myId] / N - 1; ++i) {
        for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2) {
            solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                   (solution[i * N + j - 1] + solution[i * N + j + 1] +
                                    solution[(i - 1) * N + j] +
                                    solution[(i + 1) * N + j])) / multiplayer;
        }
    }
}

void
Helmholtz::redAndBlackISendIRecv(vector<double> &solution, vector<double> &tempSolution, vector<int> &elementNumber,
                                 const int myId,
                                 const int np, int &shift) {
    MPI_Request req_send_up, req_recv_up, req_send_down, req_recv_down;

    if (myId != np - 1) {
        MPI_Isend(tempSolution.data() + elementNumber[myId] - 2 * N, N, MPI_DOUBLE, myId + 1, 5,
                  MPI_COMM_WORLD,
                  &req_send_up);
        MPI_Irecv(tempSolution.data() + elementNumber[myId] - N, N, MPI_DOUBLE, myId + 1, 6, MPI_COMM_WORLD,
                  &req_recv_up);
    }
    if (myId != 0) {
        MPI_Irecv(tempSolution.data(), N, MPI_DOUBLE, myId - 1, 5, MPI_COMM_WORLD, &req_recv_down);
        MPI_Isend(tempSolution.data() + N, N, MPI_DOUBLE, myId - 1, 6, MPI_COMM_WORLD, &req_send_down);
    }

    for (int i = 2; i < elementNumber[myId] / N - 2; ++i) {
        for (int j = ((i + shift) % 2) + 1; j < N - 1; j += 2) {
            solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                   (tempSolution[i * N + j - 1] + tempSolution[i * N + j + 1] +
                                    tempSolution[(i - 1) * N + j] +
                                    tempSolution[(i + 1) * N + j])) / multiplayer;
        }
    }

    if (myId != 0) {
        MPI_Wait(&req_recv_down, MPI_STATUSES_IGNORE);
    }
    if (myId != np - 1) {
        MPI_Wait(&req_recv_up, MPI_STATUSES_IGNORE);
    }

    int index = 1;
    for (int j = ((index + shift) % 2) + 1; j < N - 1; ++j) {
        solution[index * N + j] = (h * h * rightSideFunction((index + shift) * h, j * h) +
                                   (tempSolution[index * N + j - 1] + tempSolution[index * N + j + 1] +
                                    tempSolution[(index - 1) * N + j] +
                                    tempSolution[(index + 1) * N + j])) / multiplayer;
    }

    index = elementNumber[myId] / N - 2;
    for (int j = ((index + shift) % 2) + 1; j < N - 1; ++j) {
        solution[index * N + j] = (h * h * rightSideFunction((index + shift) * h, j * h) +
                                   (tempSolution[index * N + j - 1] + tempSolution[index * N + j + 1] +
                                    tempSolution[(index - 1) * N + j] +
                                    tempSolution[(index + 1) * N + j])) / multiplayer;
    }

    if (myId != np - 1) {
        MPI_Isend(solution.data() + elementNumber[myId] - 2 * N, N, MPI_DOUBLE, myId + 1, 5, MPI_COMM_WORLD,
                  &req_send_up);
        MPI_Irecv(solution.data() + elementNumber[myId] - N, N, MPI_DOUBLE, myId + 1, 6, MPI_COMM_WORLD,
                  &req_recv_up);
    }
    if (myId != 0) {
        MPI_Irecv(solution.data(), N, MPI_DOUBLE, myId - 1, 5, MPI_COMM_WORLD, &req_recv_down);
        MPI_Isend(solution.data() + N, N, MPI_DOUBLE, myId - 1, 6, MPI_COMM_WORLD, &req_send_down);
    }

    for (int i = 2; i < elementNumber[myId] / N - 2; ++i) {
        for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2) {
            solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                   (solution[i * N + j - 1] + solution[i * N + j + 1] +
                                    solution[(i - 1) * N + j] +
                                    solution[(i + 1) * N + j])) / multiplayer;
        }
    }

    if (myId != 0) {
        MPI_Wait(&req_recv_down, MPI_STATUSES_IGNORE);
    }
    if (myId != np - 1) {
        MPI_Wait(&req_recv_up, MPI_STATUSES_IGNORE);
    }

    int i = 1;
    for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2) {
        solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                               (solution[i * N + j - 1] + solution[i * N + j + 1] +
                                solution[(i - 1) * N + j] + solution[(i + 1) * N + j])) /
                              multiplayer;
    }

    i = elementNumber[myId] / N - 2;
    for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2) {
        solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                               (solution[i * N + j - 1] + solution[i * N + j + 1] +
                                solution[(i - 1) * N + j] + solution[(i + 1) * N + j])) /
                              multiplayer;
    }
}

double Helmholtz::rightSideFunction(double x, double solution) {
    return 2 * sin(M_PI * solution) + k * k * (1 - x) * x * sin(M_PI * solution) +
           M_PI * M_PI * (1 - x) * x * sin(M_PI * solution);
}


double Helmholtz::norm(const vector<double> &firstVector, const vector<double> &secondVector, const int startIndex,
                       const int endIndex) {
    double normValue = 0.0;
    for (int i = startIndex; i < endIndex; ++i) {
        normValue = max(normValue, abs(firstVector[i] - secondVector[i]));
    }
    return normValue;
}

void
Helmholtz::gatherSolution(vector<int> &numOfElement, vector<double> &tempSolution, vector<double> &solution,
                          vector<int> &displacementOfElement, const int np,
                          const int myId) {
    int size;
    if ((myId == 0 || myId == np - 1) && np != 1) {
        size = numOfElement[myId] - N;
    } else if (np != 1) {
        size = numOfElement[myId] - 2 * N;
    } else {
        size = numOfElement[myId];
    }

    MPI_Gatherv((myId == 0) ? tempSolution.data() : tempSolution.data() + N, size, MPI_DOUBLE, solution.data(),
                numOfElement.data(),
                displacementOfElement.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void Helmholtz::preciseSolution(vector<double> &preciseVectorSolution) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            preciseVectorSolution[i * N + j] = (1 - i * h) * i * h * sin(M_PI * j * h);
        }
    }
}
