#include "jacobi.h"
#include "const.h"
#include "shared.h"
#include <mpi.h>


void JacobiSendRecv(const int myId,
                    const int numOfProcessors) {
    vector<int> str_per_proc, nums_start;
    int str_local, nums_local;
    double norm_local, norm_err;

    str_split(myId, numOfProcessors, str_local, nums_local, str_per_proc, nums_start);

    vector<double> y_local(str_local * n);
    vector<double> y_next_top(n);
    vector<double> y_prev_low(n);

    int source_proc = myId ? myId - 1 : numOfProcessors - 1; // myId == 0 = numOfProcessors - 1
    int dest_proc = (myId != (numOfProcessors - 1)) ? myId + 1 : 0; // у myId == numOfProcessors - 1 = 0

    int scount = (myId != (numOfProcessors - 1)) ? n : 0; // у myId == numOfProcessors - 1 = 0
    int rcount = myId ? n : 0; // myId == 0 = 0

    vector<double> y;
    if (myId == 0)
        y.resize(n * n);

    double t1 = -MPI_Wtime();

    int iterations = 0;
    vector<double> temp(y_local.size());
    while (norm_err < COMPARE_RATE) {
        iterations++;
        std::swap(temp, y_local);

        // пересылаем нижние строки всеми процессами кроме последнего
        MPI_Send(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE, dest_proc, 42, MPI_COMM_WORLD);
        MPI_Recv(y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        // пересылаем верхние строки всеми процессами кроме нулевого
        MPI_Send(temp.data(), rcount, MPI_DOUBLE, source_proc, 46, MPI_COMM_WORLD);
        MPI_Recv(y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 46, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        /* пересчитываем все строки в полосе кроме верхней и нижней */
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = 1; j < n - 1; ++j)
                y_local[i * n + j] =
                        (temp[(i + 1) * n + j] + temp[(i - 1) * n + j] + temp[i * n + (j + 1)] + temp[i * n + (j - 1)] +
                         pow(h, 2) * right_part((nums_local + i) * h, j * h)) * yMultiplayer;

        /* пересчитываем верхние строки */
        if (myId != 0)
            for (int j = 1; j < n - 1; ++j)
                y_local[j] = (temp[n + j] + y_prev_low[j] + temp[j + 1] + temp[j - 1] +
                              pow(h, 2) * right_part(nums_local * h, j * h)) * yMultiplayer;

        /* пересчитываем нижние строки */
        if (myId != numOfProcessors - 1)
            for (int j = 1; j < n - 1; ++j)
                y_local[(str_local - 1) * n + j] =
                        (y_next_top[j] + temp[(str_local - 2) * n + j] + temp[(str_local - 1) * n + (j + 1)] +
                         temp[(str_local - 1) * n + (j - 1)] +
                         pow(h, 2) * right_part((nums_local + (str_local - 1)) * h, j * h)) * yMultiplayer;

        norm_local = proverka(temp, y_local);

        MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    }

    t1 += MPI_Wtime();

    for (int i = 0; i < numOfProcessors; ++i) {
        str_per_proc[i] *= n;
        nums_start[i] *= n;
    }

    MPI_Gatherv(y_local.data(), str_local * n, MPI_DOUBLE, y.data(), str_per_proc.data(), nums_start.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (myId == 0) {
        // точное решение
        vector<double> analitical_sol(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                analitical_sol[i * n + j] = u_exact(i * h, j * h);

        std::cout << "\n\n 1. Send + Recv";
        std::cout << "\n\t norm: " << proverka(y, analitical_sol);
        std::cout << "\n\t iterations: " << iterations;
        printf("\n\t time: %.4f", t1);
    }
}

void JacobiSendAndRecv(const int myId,
                       const int numOfProcessors) {

    vector<int> str_per_proc, nums_start;
    int str_local, nums_local;
    double norm_local, norm_err;

    str_split(myId, numOfProcessors, str_local, nums_local, str_per_proc, nums_start);

    vector<double> y_local(str_local * n);
    vector<double> y_next_top(n);
    vector<double> y_prev_low(n);

    vector<double> y;
    if (myId == 0)
        y.resize(n * n);

    double t1 = -MPI_Wtime();

    int source_proc = myId ? myId - 1 : numOfProcessors - 1;
    int dest_proc = (myId != (numOfProcessors - 1)) ? myId + 1 : 0;

    int scount = (myId != (numOfProcessors - 1)) ? n : 0;
    int rcount = myId ? n : 0;

    int iterations = 0;
    vector<double> temp(y_local.size());
    while (norm_err < COMPARE_RATE) {
        iterations++;

        std::swap(temp, y_local);

        // пересылаем нижние и верхние строки
        MPI_Sendrecv(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE, dest_proc, 42, y_prev_low.data(), rcount,
                     MPI_DOUBLE, source_proc, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        MPI_Sendrecv(temp.data(), rcount, MPI_DOUBLE, source_proc, 46, y_next_top.data(), scount, MPI_DOUBLE, dest_proc,
                     46, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        /* пересчитываем все строки в полосе кроме верхней и нижней */
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = 1; j < n - 1; ++j)
                y_local[i * n + j] =
                        (temp[(i + 1) * n + j] + temp[(i - 1) * n + j] + temp[i * n + (j + 1)] + temp[i * n + (j - 1)] +
                         pow(h, 2) * right_part((nums_local + i) * h, j * h)) * yMultiplayer;

        /* пересчитываем верхние строки */
        if (myId != 0)
            for (int j = 1; j < n - 1; ++j)
                y_local[j] = (temp[n + j] + y_prev_low[j] + temp[j + 1] + temp[j - 1] +
                              pow(h, 2) * right_part(nums_local * h, j * h)) * yMultiplayer;

        /* пересчитываем нижние строки */
        if (myId != numOfProcessors - 1)
            for (int j = 1; j < n - 1; ++j)
                y_local[(str_local - 1) * n + j] =
                        (y_next_top[j] + temp[(str_local - 2) * n + j] + temp[(str_local - 1) * n + (j + 1)] +
                         temp[(str_local - 1) * n + (j - 1)] +
                         pow(h, 2) * right_part((nums_local + (str_local - 1)) * h, j * h)) * yMultiplayer;

        norm_local = proverka(temp, y_local);

        MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    }

    t1 += MPI_Wtime();

    for (int i = 0; i < numOfProcessors; ++i) {
        str_per_proc[i] *= n;
        nums_start[i] *= n;
    }

    MPI_Gatherv(y_local.data(), str_local * n, MPI_DOUBLE, y.data(), str_per_proc.data(), nums_start.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (myId == 0) {
        // точное решение
        vector<double> analitical_sol(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                analitical_sol[i * n + j] = u_exact(i * h, j * h);

        std::cout << "\n\n 2. SendRecv";
        std::cout << "\n\t norm: " << proverka(y, analitical_sol);
        std::cout << "\n\t iterations: " << iterations;
        printf("\n\t time: %.4f", t1);
    }
}

void JacobiISendIRecv(const int myId,
                      const int numOfProcessors) {
    MPI_Request *send_req1;
    MPI_Request *send_req2;
    MPI_Request *recv_req1;
    MPI_Request *recv_req2;

    vector<int> str_per_proc, nums_start;
    int str_local, nums_local;
    double norm_local, norm_err;

    str_split(myId, numOfProcessors, str_local, nums_local, str_per_proc, nums_start);

    send_req1 = new MPI_Request[2], recv_req1 = new MPI_Request[2];
    send_req2 = new MPI_Request[2], recv_req2 = new MPI_Request[2];

    vector<double> y_local(str_local * n);
    vector<double> y_next_top(n);
    vector<double> y_prev_low(n);

    int source_proc = myId ? myId - 1 : numOfProcessors - 1; // myId == 0 = numOfProcessors - 1
    int dest_proc = (myId != (numOfProcessors - 1)) ? myId + 1 : 0; // у myId == numOfProcessors - 1 = 0

    int scount = (myId != (numOfProcessors - 1)) ? n : 0; // у myId == numOfProcessors - 1 = 0
    int rcount = myId ? n : 0; // myId == 0 = 0

    vector<double> temp(y_local.size());

    // пересылаем верхние и нижние строки temp
    MPI_Send_init(temp.data(), rcount, MPI_DOUBLE, source_proc, 0, MPI_COMM_WORLD, send_req1);
    MPI_Recv_init(y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 1, MPI_COMM_WORLD, recv_req1);

    MPI_Send_init(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE, dest_proc, 1, MPI_COMM_WORLD, send_req1 + 1);
    MPI_Recv_init(y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 0, MPI_COMM_WORLD, recv_req1 + 1);

    // пересылаем верхние и нижние строки y_local
    MPI_Send_init(y_local.data(), rcount, MPI_DOUBLE, source_proc, 0, MPI_COMM_WORLD, send_req2);
    MPI_Recv_init(y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 1, MPI_COMM_WORLD, recv_req2);

    MPI_Send_init(y_local.data() + (str_local - 1) * n, scount, MPI_DOUBLE, dest_proc, 1, MPI_COMM_WORLD,
                  send_req2 + 1);
    MPI_Recv_init(y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 0, MPI_COMM_WORLD, recv_req2 + 1);

    vector<double> y;
    if (myId == 0)
        y.resize(n * n);

    double t1 = -MPI_Wtime();

    int iterations = 0;
    while (norm_err < COMPARE_RATE) {
        iterations++;

        std::swap(temp, y_local);

        //// пересылаем верхние строки
        //MPI_Isend(temp.data(), rcount, MPI_DOUBLE, source_proc, 0, MPI_COMM_WORLD, send_req1);
        //MPI_Irecv(y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 1, MPI_COMM_WORLD, recv_req1 + 1);

        //// пересылаем нижние строки
        //MPI_Isend(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE, dest_proc, 1, MPI_COMM_WORLD, send_req1 + 1);
        //MPI_Irecv(y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 0, MPI_COMM_WORLD, recv_req1);

        if (iterations % 2 == 0) {
            MPI_Startall(2, send_req1);
            MPI_Startall(2, recv_req1);
        } else {
            MPI_Startall(2, send_req2);
            MPI_Startall(2, recv_req2);
        }

        /* пересчитываем все строки в полосе кроме верхней и нижней пока идёт пересылка */
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = 1; j < n - 1; ++j)
                y_local[i * n + j] =
                        (temp[(i + 1) * n + j] + temp[(i - 1) * n + j] + temp[i * n + (j + 1)] + temp[i * n + (j - 1)] +
                         pow(h, 2) * right_part((nums_local + i) * h, j * h)) * yMultiplayer;

        if (iterations % 2 == 0) {
            MPI_Waitall(2, send_req1, MPI_STATUSES_IGNORE);
            MPI_Waitall(2, recv_req1, MPI_STATUSES_IGNORE);
        } else {
            MPI_Waitall(2, send_req2, MPI_STATUSES_IGNORE);
            MPI_Waitall(2, recv_req2, MPI_STATUSES_IGNORE);
        }

        /* пересчитываем верхние строки */
        if (myId != 0)
            for (int j = 1; j < n - 1; ++j)
                y_local[j] = (temp[n + j] + y_prev_low[j] + temp[j + 1] + temp[j - 1] +
                              pow(h, 2) * right_part(nums_local * h, j * h)) * yMultiplayer;

        /* пересчитываем нижние строки */
        if (myId != numOfProcessors - 1)
            for (int j = 1; j < n - 1; ++j)
                y_local[(str_local - 1) * n + j] =
                        (y_next_top[j] + temp[(str_local - 2) * n + j] + temp[(str_local - 1) * n + (j + 1)] +
                         temp[(str_local - 1) * n + (j - 1)] +
                         pow(h, 2) * right_part((nums_local + (str_local - 1)) * h, j * h)) * yMultiplayer;

        norm_local = proverka(temp, y_local);

        MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }

    t1 += MPI_Wtime();

    for (int i = 0; i < numOfProcessors; ++i) {
        str_per_proc[i] *= n;
        nums_start[i] *= n;
    }

    MPI_Gatherv(y_local.data(), str_local * n, MPI_DOUBLE, y.data(), str_per_proc.data(), nums_start.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (myId == 0) {
        // точное решение
        vector<double> analitical_sol(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                analitical_sol[i * n + j] = u_exact(i * h, j * h);

        std::cout << "\n\n 3. Isend + Irecv";
        std::cout << "\n\t norm: " << proverka(y, analitical_sol);
        std::cout << "\n\t iterations: " << iterations;
        printf("\n\t time: %.4f", t1);
    }

    delete[] send_req1;
    delete[] recv_req1;
    delete[] send_req2;
    delete[] recv_req2;
}