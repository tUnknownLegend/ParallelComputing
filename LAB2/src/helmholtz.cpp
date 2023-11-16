#include "helmholtz.h"
#include "shared.h"
#include <mpi.h>
#include <iostream>

using std::vector;
using std::pair;
using std::pow;


double Helmholtz::f(double x, double y) {
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
Helmholtz::generalY(vector<int> &numOfElement, vector<double> &y_n, vector<double> &y,
                    std::vector<int> &displs, const int np,
                    const int myid) {
    int size;
    if ((myid == 0 || myid == np - 1) && np != 1)
        size = numOfElement[myid] - N;
    else if (np != 1)
        size = numOfElement[myid] - 2 * N;
    else
        size = numOfElement[myid];

    MPI_Gatherv((myid == 0) ? y_n.data() : y_n.data() + N, size, MPI_DOUBLE, y.data(), numOfElement.data(),
                displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void Helmholtz::analyt_sol(std::vector<double> &solution) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            solution[i * N + j] = (1 - i * h) * i * h * sin(M_PI * j * h);
        }
    }
}

double
Helmholtz::Jacobi(std::vector<double> &y, std::vector<double> &y_n, std::vector<int> &el_num, int myid, int np,
                  int &iterations,
                  int send_type) {
    double normValue;
    if (np == 1) {

        iterations = 0;
        do {
            ++iterations;
            for (int i = 1; i < N - 1; ++i)
                for (int j = 1; j < N - 1; ++j)
                    y[i * N + j] = (h * h * f(i * h, j * h) +
                                    (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] +
                                     y_n[(i + 1) * N + j])) / multiplayer;

            normValue = norm(y, y_n, 0, N * N);
            y_n.swap(y);
        } while (normValue > COMPARE_RATE);
    }
    if (np > 1) {
        double norma;

        int shift = 0;
        for (int i = 0; i < myid; ++i)
            shift += el_num[i] / N;
        shift -= (myid == 0) ? 0 : myid * 2;

        iterations = 0;
        do {
            if (send_type == 1) {
                MPI_Send(y_n.data() + el_num[myid] - 2 * N, (myid != np - 1) ? N : 0, MPI_DOUBLE,
                         (myid != np - 1) ? myid + 1 : 0, 1, MPI_COMM_WORLD);
                MPI_Recv(y_n.data(), (myid != 0) ? N : 0, MPI_DOUBLE, (myid != 0) ? myid - 1 : np - 1, 1,
                         MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                MPI_Send(y_n.data() + N, (myid != 0) ? N : 0, MPI_DOUBLE, (myid != 0) ? myid - 1 : np - 1, 2,
                         MPI_COMM_WORLD);
                MPI_Recv(y_n.data() + el_num[myid] - N, (myid != np - 1) ? N : 0, MPI_DOUBLE,
                         (myid != np - 1) ? myid + 1 : 0, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }
            if (send_type == 2) {
                MPI_Sendrecv(y_n.data() + el_num[myid] - 2 * N, (myid != np - 1) ? N : 0, MPI_DOUBLE,
                             (myid != np - 1) ? myid + 1 : 0, 3, y_n.data(), (myid != 0) ? N : 0, MPI_DOUBLE,
                             (myid != 0) ? myid - 1 : np - 1, 3, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                MPI_Sendrecv(y_n.data() + N, (myid != 0) ? N : 0, MPI_DOUBLE, (myid != 0) ? myid - 1 : np - 1, 4,
                             y_n.data() + el_num[myid] - N, (myid != np - 1) ? N : 0, MPI_DOUBLE,
                             (myid != np - 1) ? myid + 1 : 0, 4, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

            }
            MPI_Request req_send_up, req_recv_up, req_send_down, req_recv_down;
            if (send_type == 3) {
                if (myid != np - 1) {
                    MPI_Isend(y_n.data() + el_num[myid] - 2 * N, N, MPI_DOUBLE, myid + 1, 5, MPI_COMM_WORLD,
                              &req_send_up);

                    MPI_Irecv(y_n.data() + el_num[myid] - N, N, MPI_DOUBLE, myid + 1, 6, MPI_COMM_WORLD, &req_recv_up);
                }
                if (myid != 0) {
                    MPI_Irecv(y_n.data(), N, MPI_DOUBLE, myid - 1, 5, MPI_COMM_WORLD, &req_recv_down);

                    MPI_Isend(y_n.data() + N, N, MPI_DOUBLE, myid - 1, 6, MPI_COMM_WORLD, &req_send_down);
                }
            }

            ++iterations;

            if (send_type == 1 || send_type == 2) {
                for (int i = 1; i < el_num[myid] / N - 1; ++i)
                    for (int j = 1; j < N - 1; ++j)
                        y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                                        (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] +
                                         y_n[(i + 1) * N + j])) / multiplayer;
            }
            if (send_type == 3) {
                for (int i = 2; i < el_num[myid] / N - 2; ++i)
                    for (int j = 1; j < N - 1; ++j)
                        y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                                        (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] +
                                         y_n[(i + 1) * N + j])) / multiplayer;

                if (myid != 0)
                    MPI_Wait(&req_recv_down, MPI_STATUSES_IGNORE);
                if (myid != np - 1)
                    MPI_Wait(&req_recv_up, MPI_STATUSES_IGNORE);

                int i = 1;
                for (int j = 1; j < N - 1; ++j)
                    y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                                    (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] +
                                     y_n[(i + 1) * N + j])) / multiplayer;

                i = el_num[myid] / N - 2;
                for (int j = 1; j < N - 1; ++j)
                    y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                                    (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] +
                                     y_n[(i + 1) * N + j])) / multiplayer;
            }

            norma = norm(y, y_n, (myid == 0) ? 0 : N, (myid == np) ? el_num[myid] : el_num[myid] - N);
            MPI_Allreduce(&norma, &normValue, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            y_n.swap(y);
        } while (normValue > COMPARE_RATE);
    }
    if (myid == 0) {
        if (send_type == 1) {
            std::cout << "Jacobi" << " (MPI_Send + MPI_Recv)\n";
        } else if (send_type == 2) {
            std::cout << "Jacobi" << " (MPI_SendRecv)\n";
        } else if (send_type == 3) {
            std::cout << "Jacobi" << " (MPI_ISend + MPI_IRecv)\n";
        }
    }
    return normValue;
}

double
Helmholtz::Zeidel(std::vector<double> &y, std::vector<double> &y_n, std::vector<int> el_num, int myid, int np,
                  int &iterations,
                  int send_type) {
    double normValue;
    if (np == 1) {
        iterations = 0;
        do {
            ++iterations;
            for (int i = 1; i < N - 1; ++i)
                for (int j = (i % 2) + 1; j < N - 1; j += 2)
                    y[i * N + j] = (h * h * f(i * h, j * h) +
                                    (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] +
                                     y_n[(i + 1) * N + j])) / multiplayer;

            for (int i = 1; i < N - 1; ++i)
                for (int j = ((i + 1) % 2) + 1; j < N - 1; j += 2)
                    y[i * N + j] = (h * h * f(i * h, j * h) +
                                    (y[i * N + j - 1] + y[i * N + j + 1] + y[(i - 1) * N + j] + y[(i + 1) * N + j])) /
                                   multiplayer;

            normValue = norm(y, y_n, 0, N * N);
            y_n.swap(y);
        } while (normValue > COMPARE_RATE);
    }
    if (np > 1) {
        double norma;

        int shift = 0;
        for (int i = 0; i < myid; ++i)
            shift += el_num[i] / N;
        shift -= (myid == 0) ? 0 : myid * 2;

        iterations = 0;
        do {
            if (send_type == 1) {
                MPI_Send(y_n.data() + el_num[myid] - 2 * N, (myid != np - 1) ? N : 0, MPI_DOUBLE,
                         (myid != np - 1) ? myid + 1 : 0, 1, MPI_COMM_WORLD);
                MPI_Recv(y_n.data(), (myid != 0) ? N : 0, MPI_DOUBLE, (myid != 0) ? myid - 1 : np - 1, 1,
                         MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                MPI_Send(y_n.data() + N, (myid != 0) ? N : 0, MPI_DOUBLE, (myid != 0) ? myid - 1 : np - 1, 2,
                         MPI_COMM_WORLD);
                MPI_Recv(y_n.data() + el_num[myid] - N, (myid != np - 1) ? N : 0, MPI_DOUBLE,
                         (myid != np - 1) ? myid + 1 : 0, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }
            if (send_type == 2) {
                MPI_Sendrecv(y_n.data() + el_num[myid] - 2 * N, (myid != np - 1) ? N : 0, MPI_DOUBLE,
                             (myid != np - 1) ? myid + 1 : 0, 3, y_n.data(), (myid != 0) ? N : 0, MPI_DOUBLE,
                             (myid != 0) ? myid - 1 : np - 1, 3, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                MPI_Sendrecv(y_n.data() + N, (myid != 0) ? N : 0, MPI_DOUBLE, (myid != 0) ? myid - 1 : np - 1, 4,
                             y_n.data() + el_num[myid] - N, (myid != np - 1) ? N : 0, MPI_DOUBLE,
                             (myid != np - 1) ? myid + 1 : 0, 4, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }
            MPI_Request req_send_up, req_recv_up, req_send_down, req_recv_down;
            if (send_type == 3) {

                if (myid != np - 1) {
                    MPI_Isend(y_n.data() + el_num[myid] - 2 * N, N, MPI_DOUBLE, myid + 1, 5, MPI_COMM_WORLD,
                              &req_send_up);
                    MPI_Irecv(y_n.data() + el_num[myid] - N, N, MPI_DOUBLE, myid + 1, 6, MPI_COMM_WORLD, &req_recv_up);
                }
                if (myid != 0) {
                    MPI_Irecv(y_n.data(), N, MPI_DOUBLE, myid - 1, 5, MPI_COMM_WORLD, &req_recv_down);
                    MPI_Isend(y_n.data() + N, N, MPI_DOUBLE, myid - 1, 6, MPI_COMM_WORLD, &req_send_down);
                }
            }

            ++iterations;


            switch (send_type) {
                case 1:
                case 2: {
                    for (int i = 1; i < el_num[myid] / N - 1; ++i)
                        for (int j = ((i + shift) % 2) + 1; j < N - 1; j += 2)
                            y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                                            (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] +
                                             y_n[(i + 1) * N + j])) / multiplayer;
                }
                    break;
                case 3: {
                    for (int i = 2; i < el_num[myid] / N - 2; ++i)
                        for (int j = ((i + shift) % 2) + 1; j < N - 1; j += 2)
                            y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                                            (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] +
                                             y_n[(i + 1) * N + j])) / multiplayer;

                    if (myid != 0)
                        MPI_Wait(&req_recv_down, MPI_STATUSES_IGNORE);
                    if (myid != np - 1)
                        MPI_Wait(&req_recv_up, MPI_STATUSES_IGNORE);

                    int i = 1;
                    for (int j = ((i + shift) % 2) + 1; j < N - 1; ++j)
                        y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                                        (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] +
                                         y_n[(i + 1) * N + j])) / multiplayer;

                    i = el_num[myid] / N - 2;
                    for (int j = ((i + shift) % 2) + 1; j < N - 1; ++j)
                        y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                                        (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] +
                                         y_n[(i + 1) * N + j])) / multiplayer;
                }
                    break;
                default:
                    std::cerr << "method not implemented";
            }

            switch (send_type) {
                case 1: {
                    MPI_Send(y.data() + el_num[myid] - 2 * N, (myid != np - 1) ? N : 0, MPI_DOUBLE,
                             (myid != np - 1) ? myid + 1 : 0, 1, MPI_COMM_WORLD);
                    MPI_Recv(y.data(), (myid != 0) ? N : 0, MPI_DOUBLE, (myid != 0) ? myid - 1 : np - 1, 1,
                             MPI_COMM_WORLD,
                             MPI_STATUSES_IGNORE);

                    MPI_Send(y.data() + N, (myid != 0) ? N : 0, MPI_DOUBLE, (myid != 0) ? myid - 1 : np - 1, 2,
                             MPI_COMM_WORLD);
                    MPI_Recv(y.data() + el_num[myid] - N, (myid != np - 1) ? N : 0, MPI_DOUBLE,
                             (myid != np - 1) ? myid + 1 : 0, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
                }
                    break;
                case 2: {
                    MPI_Sendrecv(y.data() + el_num[myid] - 2 * N, (myid != np - 1) ? N : 0, MPI_DOUBLE,
                                 (myid != np - 1) ? myid + 1 : 0, 3, y.data(), (myid != 0) ? N : 0, MPI_DOUBLE,
                                 (myid != 0) ? myid - 1 : np - 1, 3, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                    MPI_Sendrecv(y.data() + N, (myid != 0) ? N : 0, MPI_DOUBLE, (myid != 0) ? myid - 1 : np - 1, 4,
                                 y.data() + el_num[myid] - N, (myid != np - 1) ? N : 0, MPI_DOUBLE,
                                 (myid != np - 1) ? myid + 1 : 0, 4, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
                }
                    break;
                case 3: {
                    if (myid != np - 1) {
                        MPI_Isend(y.data() + el_num[myid] - 2 * N, N, MPI_DOUBLE, myid + 1, 5, MPI_COMM_WORLD,
                                  &req_send_up);
                        MPI_Irecv(y.data() + el_num[myid] - N, N, MPI_DOUBLE, myid + 1, 6, MPI_COMM_WORLD,
                                  &req_recv_up);
                    }
                    if (myid != 0) {
                        MPI_Irecv(y.data(), N, MPI_DOUBLE, myid - 1, 5, MPI_COMM_WORLD, &req_recv_down);
                        MPI_Isend(y.data() + N, N, MPI_DOUBLE, myid - 1, 6, MPI_COMM_WORLD, &req_send_down);
                    }
                }
                    break;
                default:
                    std::cerr << "method not implemented";
            }

            int initialI = (send_type == 3 ? 2 : 1);
            for (int i = initialI; i < el_num[myid] / N - 2; ++i)
                for (int j = (((i + shift) + 1) % 2) + 1; j < N - initialI; j += 2)
                    y[i * N + j] = (pow(h, 2) * f((i + shift) * h, j * h) +
                                    (y[i * N + j - 1] + y[i * N + j + 1] + y[(i - 1) * N + j] +
                                     y[(i + 1) * N + j])) / multiplayer;

            if (send_type == 3) {
                if (myid != 0)
                    MPI_Wait(&req_recv_down, MPI_STATUSES_IGNORE);
                if (myid != np - 1)
                    MPI_Wait(&req_recv_up, MPI_STATUSES_IGNORE);

                int i = 1;
                for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2)
                    y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                                    (y[i * N + j - 1] + y[i * N + j + 1] + y[(i - 1) * N + j] + y[(i + 1) * N + j])) /
                                   multiplayer;

                i = el_num[myid] / N - 2;
                for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2)
                    y[i * N + j] = (h * h * f((i + shift) * h, j * h) +
                                    (y[i * N + j - 1] + y[i * N + j + 1] + y[(i - 1) * N + j] + y[(i + 1) * N + j])) /
                                   multiplayer;

            }

            norma = norm(y, y_n, (myid == 0) ? 0 : N, (myid == np) ? el_num[myid] : el_num[myid] - N);
            MPI_Allreduce(&norma, &normValue, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            y_n.swap(y);
        } while (normValue > COMPARE_RATE);
    }
    if (myid == 0) {
        if (send_type == 1) {
            std::cout << "Zeidel" << " (MPI_Send + MPI_Recv)\n";
        } else if (send_type == 2) {
            std::cout << "Zeidel" << " (MPI_SendRecv)\n";
        } else if (send_type == 3) {
            std::cout << "Zeidel" << " (MPI_ISend + MPI_IRecv)\n";
        }
    }
    return normValue;
}

void Helmholtz::zero(std::vector<double> &A) {
    for (double &i: A)
        i = 0.0;
}
