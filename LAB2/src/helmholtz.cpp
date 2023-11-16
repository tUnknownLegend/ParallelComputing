#include "helmholtz.h"
#include "shared.h"
#include <mpi.h>
#include <iostream>

using std::vector;
using std::pair;
using std::pow;

double Helmholtz::rightSideFunction(double x, double y) {
    return 2 * sin(M_PI * y) + k * k * (1 - x) * x * sin(M_PI * y) + M_PI * M_PI * (1 - x) * x * sin(M_PI * y);
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
Helmholtz::gatherSolution(vector<int> &numOfElement, vector<double> &y_n, vector<double> &y,
                          std::vector<int> &displacementOfElement, const int np,
                          const int myId) {
    int size;
    if ((myId == 0 || myId == np - 1) && np != 1)
        size = numOfElement[myId] - N;
    else if (np != 1)
        size = numOfElement[myId] - 2 * N;
    else
        size = numOfElement[myId];

    MPI_Gatherv((myId == 0) ? y_n.data() : y_n.data() + N, size, MPI_DOUBLE, y.data(), numOfElement.data(),
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
Helmholtz::Jacobi(std::vector<double> &y, std::vector<double> &y_n, std::vector<int> &el_num, int myId, int np,
                  int &iterationCount,
                  const JacobiSolutionMethod methodType) {
    double normValue;
    if (np == 1) {

        iterationCount = 0;
        do {
            ++iterationCount;
            for (int i = 1; i < N - 1; ++i)
                for (int j = 1; j < N - 1; ++j)
                    y[i * N + j] = (h * h * rightSideFunction(i * h, j * h) +
                                    (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] +
                                     y_n[(i + 1) * N + j])) / multiplayer;

            normValue = norm(y, y_n, 0, N * N);
            y_n.swap(y);
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
            MPI_Request req_send_up, req_recv_up, req_send_down, req_recv_down;
            switch (methodType) {
                case JacobiSendReceive: {
                    for (int i = 1; i < el_num[myId] / N - 1; ++i)
                        for (int j = 1; j < N - 1; ++j)
                            y[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                            (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] +
                                             y_n[(i + 1) * N + j])) / multiplayer;

                    MPI_Send(y_n.data() + el_num[myId] - 2 * N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                             (myId != np - 1) ? myId + 1 : 0, 1, MPI_COMM_WORLD);
                    MPI_Recv(y_n.data(), (myId != 0) ? N : 0, MPI_DOUBLE, (myId != 0) ? myId - 1 : np - 1, 1,
                             MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                    MPI_Send(y_n.data() + N, (myId != 0) ? N : 0, MPI_DOUBLE, (myId != 0) ? myId - 1 : np - 1, 2,
                             MPI_COMM_WORLD);
                    MPI_Recv(y_n.data() + el_num[myId] - N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                             (myId != np - 1) ? myId + 1 : 0, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
                }
                    break;
                case JacobiSendAndReceive: {
                    for (int i = 1; i < el_num[myId] / N - 1; ++i)
                        for (int j = 1; j < N - 1; ++j)
                            y[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                            (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] +
                                             y_n[(i + 1) * N + j])) / multiplayer;

                    MPI_Sendrecv(y_n.data() + el_num[myId] - 2 * N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                                 (myId != np - 1) ? myId + 1 : 0, 3, y_n.data(), (myId != 0) ? N : 0, MPI_DOUBLE,
                                 (myId != 0) ? myId - 1 : np - 1, 3, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                    MPI_Sendrecv(y_n.data() + N, (myId != 0) ? N : 0, MPI_DOUBLE, (myId != 0) ? myId - 1 : np - 1, 4,
                                 y_n.data() + el_num[myId] - N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                                 (myId != np - 1) ? myId + 1 : 0, 4, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                }
                    break;
                case JacobiISendIReceive: {
                    if (myId != np - 1) {
                        MPI_Isend(y_n.data() + el_num[myId] - 2 * N, N, MPI_DOUBLE, myId + 1, 5, MPI_COMM_WORLD,
                                  &req_send_up);

                        MPI_Irecv(y_n.data() + el_num[myId] - N, N, MPI_DOUBLE, myId + 1, 6, MPI_COMM_WORLD,
                                  &req_recv_up);
                    }
                    if (myId != 0) {
                        MPI_Irecv(y_n.data(), N, MPI_DOUBLE, myId - 1, 5, MPI_COMM_WORLD, &req_recv_down);

                        MPI_Isend(y_n.data() + N, N, MPI_DOUBLE, myId - 1, 6, MPI_COMM_WORLD, &req_send_down);
                    }

                    for (int i = 2; i < el_num[myId] / N - 2; ++i)
                        for (int j = 1; j < N - 1; ++j)
                            y[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                            (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] +
                                             y_n[(i + 1) * N + j])) / multiplayer;

                    if (myId != 0)
                        MPI_Wait(&req_recv_down, MPI_STATUSES_IGNORE);
                    if (myId != np - 1)
                        MPI_Wait(&req_recv_up, MPI_STATUSES_IGNORE);

                    int i = 1;
                    for (int j = 1; j < N - 1; ++j)
                        y[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                        (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] +
                                         y_n[(i + 1) * N + j])) / multiplayer;

                    i = el_num[myId] / N - 2;
                    for (int j = 1; j < N - 1; ++j)
                        y[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                        (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] +
                                         y_n[(i + 1) * N + j])) / multiplayer;
                }
                    break;
            }

            norma = norm(y, y_n, (myId == 0) ? 0 : N, (myId == np) ? el_num[myId] : el_num[myId] - N);
            MPI_Allreduce(&norma, &normValue, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            y_n.swap(y);
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
Helmholtz::redAndBlackMethod(vector<double> &y, vector<double> &y_n, vector<int> el_num, const int myId, const int np,
                             int &iterationCount,
                             const RedAndBlackSolutionMethod methodType) {
    double normValue;
    if (np == 1) {
        iterationCount = 0;
        do {
            ++iterationCount;
            for (int i = 1; i < N - 1; ++i)
                for (int j = (i % 2) + 1; j < N - 1; j += 2)
                    y[i * N + j] = (h * h * rightSideFunction(i * h, j * h) +
                                    (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] +
                                     y_n[(i + 1) * N + j])) / multiplayer;

            for (int i = 1; i < N - 1; ++i)
                for (int j = ((i + 1) % 2) + 1; j < N - 1; j += 2)
                    y[i * N + j] = (h * h * rightSideFunction(i * h, j * h) +
                                    (y[i * N + j - 1] + y[i * N + j + 1] + y[(i - 1) * N + j] + y[(i + 1) * N + j])) /
                                   multiplayer;

            normValue = norm(y, y_n, 0, N * N);
            y_n.swap(y);
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
            if (methodType == RedAndBlackSendReceive) {
                MPI_Send(y_n.data() + el_num[myId] - 2 * N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                         (myId != np - 1) ? myId + 1 : 0, 1, MPI_COMM_WORLD);
                MPI_Recv(y_n.data(), (myId != 0) ? N : 0, MPI_DOUBLE, (myId != 0) ? myId - 1 : np - 1, 1,
                         MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                MPI_Send(y_n.data() + N, (myId != 0) ? N : 0, MPI_DOUBLE, (myId != 0) ? myId - 1 : np - 1, 2,
                         MPI_COMM_WORLD);
                MPI_Recv(y_n.data() + el_num[myId] - N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                         (myId != np - 1) ? myId + 1 : 0, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }
            if (methodType == RedAndBlackSendAndReceive) {
                MPI_Sendrecv(y_n.data() + el_num[myId] - 2 * N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                             (myId != np - 1) ? myId + 1 : 0, 3, y_n.data(), (myId != 0) ? N : 0, MPI_DOUBLE,
                             (myId != 0) ? myId - 1 : np - 1, 3, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                MPI_Sendrecv(y_n.data() + N, (myId != 0) ? N : 0, MPI_DOUBLE, (myId != 0) ? myId - 1 : np - 1, 4,
                             y_n.data() + el_num[myId] - N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                             (myId != np - 1) ? myId + 1 : 0, 4, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }
            MPI_Request req_send_up, req_recv_up, req_send_down, req_recv_down;
            if (methodType == RedAndBlackISendIReceive) {

                if (myId != np - 1) {
                    MPI_Isend(y_n.data() + el_num[myId] - 2 * N, N, MPI_DOUBLE, myId + 1, 5, MPI_COMM_WORLD,
                              &req_send_up);
                    MPI_Irecv(y_n.data() + el_num[myId] - N, N, MPI_DOUBLE, myId + 1, 6, MPI_COMM_WORLD, &req_recv_up);
                }
                if (myId != 0) {
                    MPI_Irecv(y_n.data(), N, MPI_DOUBLE, myId - 1, 5, MPI_COMM_WORLD, &req_recv_down);
                    MPI_Isend(y_n.data() + N, N, MPI_DOUBLE, myId - 1, 6, MPI_COMM_WORLD, &req_send_down);
                }
            }

            ++iterationCount;


            switch (methodType) {
                case RedAndBlackSendReceive:
                case RedAndBlackSendAndReceive: {
                    for (int i = 1; i < el_num[myId] / N - 1; ++i)
                        for (int j = ((i + shift) % 2) + 1; j < N - 1; j += 2)
                            y[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                            (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] +
                                             y_n[(i + 1) * N + j])) / multiplayer;
                }
                    break;
                case RedAndBlackISendIReceive: {
                    for (int i = 2; i < el_num[myId] / N - 2; ++i)
                        for (int j = ((i + shift) % 2) + 1; j < N - 1; j += 2)
                            y[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                            (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] +
                                             y_n[(i + 1) * N + j])) / multiplayer;

                    if (myId != 0)
                        MPI_Wait(&req_recv_down, MPI_STATUSES_IGNORE);
                    if (myId != np - 1)
                        MPI_Wait(&req_recv_up, MPI_STATUSES_IGNORE);

                    int i = 1;
                    for (int j = ((i + shift) % 2) + 1; j < N - 1; ++j)
                        y[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                        (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] +
                                         y_n[(i + 1) * N + j])) / multiplayer;

                    i = el_num[myId] / N - 2;
                    for (int j = ((i + shift) % 2) + 1; j < N - 1; ++j)
                        y[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                        (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] +
                                         y_n[(i + 1) * N + j])) / multiplayer;
                }
                    break;
                default:
                    std::cerr << "method not implemented";
            }

            switch (methodType) {
                case RedAndBlackSendReceive: {
                    MPI_Send(y.data() + el_num[myId] - 2 * N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                             (myId != np - 1) ? myId + 1 : 0, 1, MPI_COMM_WORLD);
                    MPI_Recv(y.data(), (myId != 0) ? N : 0, MPI_DOUBLE, (myId != 0) ? myId - 1 : np - 1, 1,
                             MPI_COMM_WORLD,
                             MPI_STATUSES_IGNORE);

                    MPI_Send(y.data() + N, (myId != 0) ? N : 0, MPI_DOUBLE, (myId != 0) ? myId - 1 : np - 1, 2,
                             MPI_COMM_WORLD);
                    MPI_Recv(y.data() + el_num[myId] - N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                             (myId != np - 1) ? myId + 1 : 0, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
                }
                    break;
                case RedAndBlackSendAndReceive: {
                    MPI_Sendrecv(y.data() + el_num[myId] - 2 * N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                                 (myId != np - 1) ? myId + 1 : 0, 3, y.data(), (myId != 0) ? N : 0, MPI_DOUBLE,
                                 (myId != 0) ? myId - 1 : np - 1, 3, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                    MPI_Sendrecv(y.data() + N, (myId != 0) ? N : 0, MPI_DOUBLE, (myId != 0) ? myId - 1 : np - 1, 4,
                                 y.data() + el_num[myId] - N, (myId != np - 1) ? N : 0, MPI_DOUBLE,
                                 (myId != np - 1) ? myId + 1 : 0, 4, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
                }
                    break;
                case RedAndBlackISendIReceive: {
                    if (myId != np - 1) {
                        MPI_Isend(y.data() + el_num[myId] - 2 * N, N, MPI_DOUBLE, myId + 1, 5, MPI_COMM_WORLD,
                                  &req_send_up);
                        MPI_Irecv(y.data() + el_num[myId] - N, N, MPI_DOUBLE, myId + 1, 6, MPI_COMM_WORLD,
                                  &req_recv_up);
                    }
                    if (myId != 0) {
                        MPI_Irecv(y.data(), N, MPI_DOUBLE, myId - 1, 5, MPI_COMM_WORLD, &req_recv_down);
                        MPI_Isend(y.data() + N, N, MPI_DOUBLE, myId - 1, 6, MPI_COMM_WORLD, &req_send_down);
                    }
                }
                    break;
                default:
                    std::cerr << "method not implemented";
            }

            if (methodType == RedAndBlackSendReceive || methodType == RedAndBlackSendAndReceive) {
                for (int i = 1; i < el_num[myId] / N - 1; ++i)
                    for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2)
                        y[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                        (y[i * N + j - 1] + y[i * N + j + 1] + y[(i - 1) * N + j] +
                                         y[(i + 1) * N + j])) / multiplayer;
            }

            if (methodType == RedAndBlackISendIReceive) {
                for (int i = 2; i < el_num[myId] / N - 2; ++i)
                    for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2)
                        y[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                        (y[i * N + j - 1] + y[i * N + j + 1] + y[(i - 1) * N + j] +
                                         y[(i + 1) * N + j])) / multiplayer;


                if (myId != 0)
                    MPI_Wait(&req_recv_down, MPI_STATUSES_IGNORE);
                if (myId != np - 1)
                    MPI_Wait(&req_recv_up, MPI_STATUSES_IGNORE);

                int i = 1;
                for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2)
                    y[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                    (y[i * N + j - 1] + y[i * N + j + 1] + y[(i - 1) * N + j] + y[(i + 1) * N + j])) /
                                   multiplayer;

                i = el_num[myId] / N - 2;
                for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2)
                    y[i * N + j] = (h * h * rightSideFunction((i + shift) * h, j * h) +
                                    (y[i * N + j - 1] + y[i * N + j + 1] + y[(i - 1) * N + j] + y[(i + 1) * N + j])) /
                                   multiplayer;

            }

            norma = norm(y, y_n, (myId == 0) ? 0 : N, (myId == np) ? el_num[myId] : el_num[myId] - N);
            MPI_Allreduce(&norma, &normValue, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            y_n.swap(y);
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

