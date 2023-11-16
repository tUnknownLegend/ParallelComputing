#include "helmholtz.h"
#include "shared.h"
#include <mpi.h>
#include <iostream>

using std::vector;
using std::pair;
using std::pow;

double Helmholtz::rightSideFunction(double x, double solution) {
    return 2 * sin(M_PI * solution) + k * k * (1 - x) * x * sin(M_PI * solution) +
           M_PI * M_PI * (1 - x) * x * sin(M_PI * solution);
}


double Helmholtz::norm(const vector<double> &firstVector, const vector<double> &secondVector, const int startIndex,
                       const int endIndex) {
    double normValue = 0.0;
    for (int i = startIndex; i < endIndex; ++i) {
        normValue = std::max(normValue, std::abs(firstVector[i] - secondVector[i]));
    }
    return normValue;
}

void
Helmholtz::gatherSolution(vector<int> &numOfElement, vector<double> &tempSolution, vector<double> &solution,
                          std::vector<int> &displacementOfElement, const int np,
                          const int myId) {
    int size;
    if ((myId == 0 || myId == np - 1) && np != 1)
        size = numOfElement[myId] - N;
    else if (np != 1)
        size = numOfElement[myId] - 2 * N;
    else
        size = numOfElement[myId];

    MPI_Gatherv((myId == 0) ? tempSolution.data() : tempSolution.data() + N, size, MPI_DOUBLE, solution.data(),
                numOfElement.data(),
                displacementOfElement.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void Helmholtz::preciseSolution(std::vector<double> &solution) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            solution[i * N + j] = (1 - i * h) * i * h * sin(M_PI * j * h);
        }
    }
}

double
Helmholtz::Jacobi(std::vector<double> &solution, std::vector<double> &tempSolution, std::vector<int> &el_num, int myId,
                  int np,
                  int &iterationCount,
                  const JacobiSolutionMethod methodType) {
    double normValue;
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
        double norma;

        int shift = 0;
        for (int i = 0; i < myId; ++i)
            shift += el_num[i] / N;
        shift -= (myId == 0) ? 0 : myId * 2;

        iterationCount = 0;
        do {
            ++iterationCount;
            switch (methodType) {
                case JacobiSendReceive: {
                    MPI_Send(tempSolution.data() + el_num[myId] - 2 * N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                             (myId != np - 1) ? myId + 1 : 0, 1, MPI_COMM_WORLD);
                    MPI_Recv(tempSolution.data(), (myId != 0) ? N : 0, MPI_DOUBLE, (myId != 0) ? myId - 1 : np - 1, 1,
                             MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                    MPI_Send(tempSolution.data() + N, (myId != 0) ? N : 0, MPI_DOUBLE, (myId != 0) ? myId - 1 : np - 1,
                             2,
                             MPI_COMM_WORLD);
                    MPI_Recv(tempSolution.data() + el_num[myId] - N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                             (myId != np - 1) ? myId + 1 : 0, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                    for (int i = 1; i < el_num[myId] / N - 1; ++i) {
                        for (int j = 1; j < N - 1; ++j) {
                            solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                                   (tempSolution[i * N + j - 1] + tempSolution[i * N + j + 1] +
                                                    tempSolution[(i - 1) * N + j] +
                                                    tempSolution[(i + 1) * N + j])) / multiplayer;
                        }
                    }
                }
                    break;
                case JacobiSendAndReceive: {
                    MPI_Sendrecv(tempSolution.data() + el_num[myId] - 2 * N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                                 (myId != np - 1) ? myId + 1 : 0, 3, tempSolution.data(), (myId != 0) ? N : 0,
                                 MPI_DOUBLE,
                                 (myId != 0) ? myId - 1 : np - 1, 3, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                    MPI_Sendrecv(tempSolution.data() + N, (myId != 0) ? N : 0, MPI_DOUBLE,
                                 (myId != 0) ? myId - 1 : np - 1, 4,
                                 tempSolution.data() + el_num[myId] - N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                                 (myId != np - 1) ? myId + 1 : 0, 4, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                    for (int i = 1; i < el_num[myId] / N - 1; ++i) {
                        for (int j = 1; j < N - 1; ++j) {
                            solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                                   (tempSolution[i * N + j - 1] + tempSolution[i * N + j + 1] +
                                                    tempSolution[(i - 1) * N + j] +
                                                    tempSolution[(i + 1) * N + j])) / multiplayer;
                        }
                    }
                }
                    break;
                case JacobiISendIReceive: {
                    MPI_Request req_send_up, req_recv_up, req_send_down, req_recv_down;
                    if (myId != np - 1) {
                        MPI_Isend(tempSolution.data() + el_num[myId] - 2 * N, N, MPI_DOUBLE, myId + 1, 5,
                                  MPI_COMM_WORLD,
                                  &req_send_up);

                        MPI_Irecv(tempSolution.data() + el_num[myId] - N, N, MPI_DOUBLE, myId + 1, 6, MPI_COMM_WORLD,
                                  &req_recv_up);
                    }
                    if (myId != 0) {
                        MPI_Irecv(tempSolution.data(), N, MPI_DOUBLE, myId - 1, 5, MPI_COMM_WORLD, &req_recv_down);

                        MPI_Isend(tempSolution.data() + N, N, MPI_DOUBLE, myId - 1, 6, MPI_COMM_WORLD, &req_send_down);
                    }

                    for (int i = 2; i < el_num[myId] / N - 2; ++i) {
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

                    i = el_num[myId] / N - 2;
                    for (int j = 1; j < N - 1; ++j) {
                        solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                               (tempSolution[i * N + j - 1] + tempSolution[i * N + j + 1] +
                                                tempSolution[(i - 1) * N + j] +
                                                tempSolution[(i + 1) * N + j])) / multiplayer;
                    }
                }
                    break;
            }

            norma = norm(solution, tempSolution, (myId == 0) ? 0 : N, (myId == np) ? el_num[myId] : el_num[myId] - N);
            MPI_Allreduce(&norma, &normValue, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            tempSolution.swap(solution);
        } while (normValue > COMPARE_RATE);
    }
    if (myId == 0) {
        switch (methodType) {
            case JacobiSendReceive:
                std::cout << methodType << ". JacobiSendRecv\n";
                break;
            case JacobiSendAndReceive:
                std::cout << methodType << ". JacobiSendAndRecv\n";
                break;
            case JacobiISendIReceive:
                std::cout << methodType << ". JacobiISendIRecv\n";
                break;
            default:
                std::cerr << methodType << ". method not implemented\n";
        }
    }
    return normValue;
}

double
Helmholtz::redAndBlackMethod(vector<double> &solution, vector<double> &tempSolution, vector<int> el_num, const int myId,
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
        double norma;

        int shift = 0;
        for (int i = 0; i < myId; ++i)
            shift += el_num[i] / N;
        shift -= (myId == 0) ? 0 : myId * 2;

        iterationCount = 0;
        do {
            ++iterationCount;

            switch (methodType) {
                case RedAndBlackSendReceive: {
                    MPI_Send(tempSolution.data() + el_num[myId] - 2 * N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                             (myId != np - 1) ? myId + 1 : 0, 1, MPI_COMM_WORLD);
                    MPI_Recv(tempSolution.data(), (myId != 0) ? N : 0, MPI_DOUBLE, (myId != 0) ? myId - 1 : np - 1, 1,
                             MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                    MPI_Send(tempSolution.data() + N, (myId != 0) ? N : 0, MPI_DOUBLE, (myId != 0) ? myId - 1 : np - 1,
                             2,
                             MPI_COMM_WORLD);
                    MPI_Recv(tempSolution.data() + el_num[myId] - N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                             (myId != np - 1) ? myId + 1 : 0, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                    for (int i = 1; i < el_num[myId] / N - 1; ++i) {
                        for (int j = ((i + shift) % 2) + 1; j < N - 1; j += 2) {
                            solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                                   (tempSolution[i * N + j - 1] + tempSolution[i * N + j + 1] +
                                                    tempSolution[(i - 1) * N + j] +
                                                    tempSolution[(i + 1) * N + j])) / multiplayer;
                        }
                    }

                    MPI_Send(solution.data() + el_num[myId] - 2 * N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                             (myId != np - 1) ? myId + 1 : 0, 1, MPI_COMM_WORLD);
                    MPI_Recv(solution.data(), (myId != 0) ? N : 0, MPI_DOUBLE, (myId != 0) ? myId - 1 : np - 1, 1,
                             MPI_COMM_WORLD,
                             MPI_STATUSES_IGNORE);

                    MPI_Send(solution.data() + N, (myId != 0) ? N : 0, MPI_DOUBLE, (myId != 0) ? myId - 1 : np - 1, 2,
                             MPI_COMM_WORLD);
                    MPI_Recv(solution.data() + el_num[myId] - N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                             (myId != np - 1) ? myId + 1 : 0, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                    for (int i = 1; i < el_num[myId] / N - 1; ++i) {
                        for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2) {
                            solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                                   (solution[i * N + j - 1] + solution[i * N + j + 1] +
                                                    solution[(i - 1) * N + j] +
                                                    solution[(i + 1) * N + j])) / multiplayer;
                        }
                    }
                }
                    break;
                case RedAndBlackSendAndReceive: {
                    MPI_Sendrecv(tempSolution.data() + el_num[myId] - 2 * N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                                 (myId != np - 1) ? myId + 1 : 0, 3, tempSolution.data(), (myId != 0) ? N : 0,
                                 MPI_DOUBLE,
                                 (myId != 0) ? myId - 1 : np - 1, 3, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                    MPI_Sendrecv(tempSolution.data() + N, (myId != 0) ? N : 0, MPI_DOUBLE,
                                 (myId != 0) ? myId - 1 : np - 1,
                                 4,
                                 tempSolution.data() + el_num[myId] - N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                                 (myId != np - 1) ? myId + 1 : 0, 4, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                    for (int i = 1; i < el_num[myId] / N - 1; ++i) {
                        for (int j = ((i + shift) % 2) + 1; j < N - 1; j += 2) {
                            solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                                   (tempSolution[i * N + j - 1] + tempSolution[i * N + j + 1] +
                                                    tempSolution[(i - 1) * N + j] +
                                                    tempSolution[(i + 1) * N + j])) / multiplayer;
                        }
                    }

                    MPI_Sendrecv(solution.data() + el_num[myId] - 2 * N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                                 (myId != np - 1) ? myId + 1 : 0, 3, solution.data(), (myId != 0) ? N : 0, MPI_DOUBLE,
                                 (myId != 0) ? myId - 1 : np - 1, 3, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                    MPI_Sendrecv(solution.data() + N, (myId != 0) ? N : 0, MPI_DOUBLE, (myId != 0) ? myId - 1 : np - 1,
                                 4,
                                 solution.data() + el_num[myId] - N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                                 (myId != np - 1) ? myId + 1 : 0, 4, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                    for (int i = 1; i < el_num[myId] / N - 1; ++i) {
                        for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2) {
                            solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                                   (solution[i * N + j - 1] + solution[i * N + j + 1] +
                                                    solution[(i - 1) * N + j] +
                                                    solution[(i + 1) * N + j])) / multiplayer;
                        }
                    }
                }
                    break;
                case RedAndBlackISendIReceive: {
                    MPI_Request req_send_up, req_recv_up, req_send_down, req_recv_down;

                    if (myId != np - 1) {
                        MPI_Isend(tempSolution.data() + el_num[myId] - 2 * N, N, MPI_DOUBLE, myId + 1, 5,
                                  MPI_COMM_WORLD,
                                  &req_send_up);
                        MPI_Irecv(tempSolution.data() + el_num[myId] - N, N, MPI_DOUBLE, myId + 1, 6, MPI_COMM_WORLD,
                                  &req_recv_up);
                    }
                    if (myId != 0) {
                        MPI_Irecv(tempSolution.data(), N, MPI_DOUBLE, myId - 1, 5, MPI_COMM_WORLD, &req_recv_down);
                        MPI_Isend(tempSolution.data() + N, N, MPI_DOUBLE, myId - 1, 6, MPI_COMM_WORLD, &req_send_down);
                    }

                    //

                    for (int i = 2; i < el_num[myId] / N - 2; ++i)
                        for (int j = ((i + shift) % 2) + 1; j < N - 1; j += 2)
                            solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                                   (tempSolution[i * N + j - 1] + tempSolution[i * N + j + 1] +
                                                    tempSolution[(i - 1) * N + j] +
                                                    tempSolution[(i + 1) * N + j])) / multiplayer;

                    if (myId != 0)
                        MPI_Wait(&req_recv_down, MPI_STATUSES_IGNORE);
                    if (myId != np - 1)
                        MPI_Wait(&req_recv_up, MPI_STATUSES_IGNORE);

                    int index = 1;
                    for (int j = ((index + shift) % 2) + 1; j < N - 1; ++j)
                        solution[index * N + j] = (h * h * rightSideFunction((index + shift) * h, j * h) +
                                                   (tempSolution[index * N + j - 1] + tempSolution[index * N + j + 1] +
                                                    tempSolution[(index - 1) * N + j] +
                                                    tempSolution[(index + 1) * N + j])) / multiplayer;

                    index = el_num[myId] / N - 2;
                    for (int j = ((index + shift) % 2) + 1; j < N - 1; ++j)
                        solution[index * N + j] = (h * h * rightSideFunction((index + shift) * h, j * h) +
                                                   (tempSolution[index * N + j - 1] + tempSolution[index * N + j + 1] +
                                                    tempSolution[(index - 1) * N + j] +
                                                    tempSolution[(index + 1) * N + j])) / multiplayer;

                    //

                    if (myId != np - 1) {
                        MPI_Isend(solution.data() + el_num[myId] - 2 * N, N, MPI_DOUBLE, myId + 1, 5, MPI_COMM_WORLD,
                                  &req_send_up);
                        MPI_Irecv(solution.data() + el_num[myId] - N, N, MPI_DOUBLE, myId + 1, 6, MPI_COMM_WORLD,
                                  &req_recv_up);
                    }
                    if (myId != 0) {
                        MPI_Irecv(solution.data(), N, MPI_DOUBLE, myId - 1, 5, MPI_COMM_WORLD, &req_recv_down);
                        MPI_Isend(solution.data() + N, N, MPI_DOUBLE, myId - 1, 6, MPI_COMM_WORLD, &req_send_down);
                    }

                    //

                    for (int i = 2; i < el_num[myId] / N - 2; ++i)
                        for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2)
                            solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                                   (solution[i * N + j - 1] + solution[i * N + j + 1] +
                                                    solution[(i - 1) * N + j] +
                                                    solution[(i + 1) * N + j])) / multiplayer;


                    if (myId != 0)
                        MPI_Wait(&req_recv_down, MPI_STATUSES_IGNORE);
                    if (myId != np - 1)
                        MPI_Wait(&req_recv_up, MPI_STATUSES_IGNORE);

                    int i = 1;
                    for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2)
                        solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                               (solution[i * N + j - 1] + solution[i * N + j + 1] +
                                                solution[(i - 1) * N + j] + solution[(i + 1) * N + j])) /
                                              multiplayer;

                    i = el_num[myId] / N - 2;
                    for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2)
                        solution[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                               (solution[i * N + j - 1] + solution[i * N + j + 1] +
                                                solution[(i - 1) * N + j] + solution[(i + 1) * N + j])) /
                                              multiplayer;

                    //
                }
                    break;
                default:
                    std::cerr << "method not implemented";
            }

            norma = norm(solution, tempSolution, (myId == 0) ? 0 : N, (myId == np) ? el_num[myId] : el_num[myId] - N);
            MPI_Allreduce(&norma, &normValue, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            tempSolution.swap(solution);
        } while (normValue > COMPARE_RATE);
    }
    if (myId == 0) {
        switch (methodType) {
            case RedAndBlackSendReceive:
                std::cout << methodType << ". redAndBlackMethodSendRecv\n";
                break;
            case RedAndBlackSendAndReceive:
                std::cout << methodType << ". redAndBlackMethodSendAndRecv\n";
                break;
            case RedAndBlackISendIReceive:
                std::cout << methodType << ". redAndBlackMethodISendIRecv\n";
                break;
            default:
                std::cerr << methodType << ". method not implemented\n";
        }
    }
    return normValue;
}

