#include "helmholtz.h"
#include "shared.h"
#include <mpi.h>
#include <iostream>

using std::vector;
using std::pair;
using std::pow;

Helmholtz::Helmholtz(const int inMyId,
                     const int inNumOfProcessors) {
    myId = inMyId;
    numOfProcessors = inNumOfProcessors;
}

void Helmholtz::solve(const SolutionMethod method, const std::string &name) {
    str_per_proc.resize(numOfProcessors, n / numOfProcessors);
    nums_start.resize(numOfProcessors, 0);

    for (int i = 0; i < n % numOfProcessors; ++i)
        ++str_per_proc[i];

    for (int i = 1; i < numOfProcessors; ++i)
        nums_start[i] = nums_start[i - 1] + str_per_proc[i - 1];

    MPI_Scatter(str_per_proc.data(), 1, MPI_INT, &str_local, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(nums_start.data(), 1, MPI_INT, &nums_local, 1, MPI_INT, 0, MPI_COMM_WORLD);

    sourceProcess = myId ? myId - 1 : numOfProcessors - 1; // myId == 0 = numOfProcessors - 1
    destProcess = (myId != (numOfProcessors - 1)) ? myId + 1 : 0; // у myId == numOfProcessors - 1 = 0

    scount = (myId != (numOfProcessors - 1)) ? n : 0; // у myId == numOfProcessors - 1 = 0
    rcount = myId ? n : 0; // myId == 0 = 0

    solution = vector<double>(str_local * n);
    nextTopSolution = vector<double>(n);
    prevBottomSolution = vector<double>(n);
    temp = vector<double>(solution.size());

    iterationsNum = 0;
    flag = true;

    double stopWatch = -MPI_Wtime();

    // вызов методов
    switch (method) {
        case JacobiSendReceive:
            this->calcJacobiSendReceive();
            break;
        case JacobiSendAndReceive:
            this->calcJacobiSendAndReceive();
            break;
        case JacobiISendIReceive:
            this->calcJacobiISendIReceive();
            break;
        case RedAndBlackSendReceive:
            this->calcRedAndBlackSendReceive();
            break;
        case RedAndBlackSendAndReceive:
            this->calcRedAndBlackSendAndReceive();
            break;
        case RedAndBlackISendIReceive:
            this->calcRedAndBlackISendIReceive();
            break;
        default:
            std::cerr << "method not implemented";
    }


    stopWatch += MPI_Wtime();

    for (int i = 0; i < numOfProcessors; ++i) {
        str_per_proc[i] *= n;
        nums_start[i] *= n;
    }

    if (myId == 0)
        y.resize(n * n);

    MPI_Gatherv(solution.data(), str_local * n, MPI_DOUBLE, y.data(), str_per_proc.data(), nums_start.data(),
                MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (myId == 0) {
        // точное решение
        vector<double> analitical_sol(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                analitical_sol[i * n + j] = u_exact(i * h, j * h);

        // name here
        std::cout << "\n\n" << method << ". " << name;
        std::cout << "\n\t norm: " << calcDiff(y, analitical_sol);
        std::cout << "\n\t iterationsNum: " << iterationsNum;
        printf("\n\t time: %.4f", stopWatch);
    }
}

void Helmholtz::calcJacobiSendReceive() {
    while (flag) {
        iterationsNum++;
        std::swap(temp, solution);

        // пересылаем нижние строки всеми процессами кроме последнего
        MPI_Send(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE, destProcess, 42, MPI_COMM_WORLD);
        MPI_Recv(prevBottomSolution.data(), rcount, MPI_DOUBLE, sourceProcess, 42, MPI_COMM_WORLD,
                 MPI_STATUSES_IGNORE);

        // пересылаем верхние строки всеми процессами кроме нулевого
        MPI_Send(temp.data(), rcount, MPI_DOUBLE, sourceProcess, 46, MPI_COMM_WORLD);
        MPI_Recv(nextTopSolution.data(), scount, MPI_DOUBLE, destProcess, 46, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        /* пересчитываем все строки в полосе кроме верхней и нижней */
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = 1; j < n - 1; ++j)
                solution[i * n + j] =
                        (temp[(i + 1) * n + j] + temp[(i - 1) * n + j] + temp[i * n + (j + 1)] + temp[i * n + (j - 1)] +
                         pow(h, 2) * right_part((nums_local + i) * h, j * h)) * yMultiplayer;

        /* пересчитываем верхние строки */
        if (myId != 0)
            for (int j = 1; j < n - 1; ++j)
                solution[j] = (temp[n + j] + prevBottomSolution[j] + temp[j + 1] + temp[j - 1] +
                               pow(h, 2) * right_part(nums_local * h, j * h)) * yMultiplayer;

        /* пересчитываем нижние строки */
        if (myId != numOfProcessors - 1)
            for (int j = 1; j < n - 1; ++j)
                solution[(str_local - 1) * n + j] =
                        (nextTopSolution[j] + temp[(str_local - 2) * n + j] + temp[(str_local - 1) * n + (j + 1)] +
                         temp[(str_local - 1) * n + (j - 1)] +
                         pow(h, 2) * right_part((nums_local + (str_local - 1)) * h, j * h)) * yMultiplayer;

        norm_local = calcDiff(temp, solution);

        MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (norm_err < COMPARE_RATE)
            flag = false;
    }
}

void Helmholtz::calcJacobiSendAndReceive() {
    while (flag) {
        iterationsNum++;
        std::swap(temp, solution);

        // пересылаем нижние и верхние строки
        MPI_Sendrecv(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE, destProcess, 42,
                     prevBottomSolution.data(),
                     rcount,
                     MPI_DOUBLE, sourceProcess, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        MPI_Sendrecv(temp.data(), rcount, MPI_DOUBLE, sourceProcess, 46, nextTopSolution.data(), scount, MPI_DOUBLE,
                     destProcess,
                     46, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        /* пересчитываем все строки в полосе кроме верхней и нижней */
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = 1; j < n - 1; ++j)
                solution[i * n + j] =
                        (temp[(i + 1) * n + j] + temp[(i - 1) * n + j] + temp[i * n + (j + 1)] + temp[i * n + (j - 1)] +
                         pow(h, 2) * right_part((nums_local + i) * h, j * h)) * yMultiplayer;

        /* пересчитываем верхние строки */
        if (myId != 0)
            for (int j = 1; j < n - 1; ++j)
                solution[j] = (temp[n + j] + prevBottomSolution[j] + temp[j + 1] + temp[j - 1] +
                               pow(h, 2) * right_part(nums_local * h, j * h)) * yMultiplayer;

        /* пересчитываем нижние строки */
        if (myId != numOfProcessors - 1)
            for (int j = 1; j < n - 1; ++j)
                solution[(str_local - 1) * n + j] =
                        (nextTopSolution[j] + temp[(str_local - 2) * n + j] + temp[(str_local - 1) * n + (j + 1)] +
                         temp[(str_local - 1) * n + (j - 1)] +
                         pow(h, 2) * right_part((nums_local + (str_local - 1)) * h, j * h)) * yMultiplayer;

        norm_local = calcDiff(temp, solution);

        MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (norm_err < COMPARE_RATE)
            flag = false;
    }

}

void Helmholtz::calcJacobiISendIReceive() {
    auto *send_req1 = new MPI_Request[2];
    auto *send_req2 = new MPI_Request[2];
    auto *recv_req1 = new MPI_Request[2];
    auto *recv_req2 = new MPI_Request[2];

    // пересылаем верхние и нижние строки temp
    MPI_Send_init(temp.data(), rcount, MPI_DOUBLE, sourceProcess, 0, MPI_COMM_WORLD, send_req1);
    MPI_Recv_init(prevBottomSolution.data(), rcount, MPI_DOUBLE, sourceProcess, 1, MPI_COMM_WORLD, recv_req1);

    MPI_Send_init(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE, destProcess, 1, MPI_COMM_WORLD, send_req1 + 1);
    MPI_Recv_init(nextTopSolution.data(), scount, MPI_DOUBLE, destProcess, 0, MPI_COMM_WORLD, recv_req1 + 1);

    // пересылаем верхние и нижние строки solution
    MPI_Send_init(solution.data(), rcount, MPI_DOUBLE, sourceProcess, 0, MPI_COMM_WORLD, send_req2);
    MPI_Recv_init(prevBottomSolution.data(), rcount, MPI_DOUBLE, sourceProcess, 1, MPI_COMM_WORLD, recv_req2);

    MPI_Send_init(solution.data() + (str_local - 1) * n, scount, MPI_DOUBLE, destProcess, 1, MPI_COMM_WORLD,
                  send_req2 + 1);
    MPI_Recv_init(nextTopSolution.data(), scount, MPI_DOUBLE, destProcess, 0, MPI_COMM_WORLD, recv_req2 + 1);

    while (flag) {
        iterationsNum++;

        std::swap(temp, solution);

        //// пересылаем верхние строки
        //MPI_Isend(temp.data(), rcount, MPI_DOUBLE, sourceProcess, 0, MPI_COMM_WORLD, send_req1);
        //MPI_Irecv(prevBottomSolution.data(), rcount, MPI_DOUBLE, sourceProcess, 1, MPI_COMM_WORLD, recv_req1 + 1);

        //// пересылаем нижние строки
        //MPI_Isend(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE, destProcess, 1, MPI_COMM_WORLD, send_req1 + 1);
        //MPI_Irecv(nextTopSolution.data(), scount, MPI_DOUBLE, destProcess, 0, MPI_COMM_WORLD, recv_req1);

        if (iterationsNum % 2 == 0) {
            MPI_Startall(2, send_req1);
            MPI_Startall(2, recv_req1);
        } else {
            MPI_Startall(2, send_req2);
            MPI_Startall(2, recv_req2);
        }

        /* пересчитываем все строки в полосе кроме верхней и нижней пока идёт пересылка */
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = 1; j < n - 1; ++j)
                solution[i * n + j] =
                        (temp[(i + 1) * n + j] + temp[(i - 1) * n + j] + temp[i * n + (j + 1)] + temp[i * n + (j - 1)] +
                         pow(h, 2) * right_part((nums_local + i) * h, j * h)) * yMultiplayer;

        if (iterationsNum % 2 == 0) {
            MPI_Waitall(2, send_req1, MPI_STATUSES_IGNORE);
            MPI_Waitall(2, recv_req1, MPI_STATUSES_IGNORE);
        } else {
            MPI_Waitall(2, send_req2, MPI_STATUSES_IGNORE);
            MPI_Waitall(2, recv_req2, MPI_STATUSES_IGNORE);
        }

        /* пересчитываем верхние строки */
        if (myId != 0)
            for (int j = 1; j < n - 1; ++j)
                solution[j] = (temp[n + j] + prevBottomSolution[j] + temp[j + 1] + temp[j - 1] +
                               pow(h, 2) * right_part(nums_local * h, j * h)) * yMultiplayer;

        /* пересчитываем нижние строки */
        if (myId != numOfProcessors - 1)
            for (int j = 1; j < n - 1; ++j)
                solution[(str_local - 1) * n + j] =
                        (nextTopSolution[j] + temp[(str_local - 2) * n + j] + temp[(str_local - 1) * n + (j + 1)] +
                         temp[(str_local - 1) * n + (j - 1)] +
                         pow(h, 2) * right_part((nums_local + (str_local - 1)) * h, j * h)) * yMultiplayer;

        norm_local = calcDiff(temp, solution);

        MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (norm_err < COMPARE_RATE)
            flag = false;
    }
    delete[] send_req1;
    delete[] recv_req1;
    delete[] send_req2;
    delete[] recv_req2;
}

void Helmholtz::calcRedAndBlackSendReceive() {
    while (flag) {
        iterationsNum++;

        std::swap(temp, solution);

        // пересылаем нижние строки всеми процессами кроме последнего
        MPI_Send(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE, destProcess, 42, MPI_COMM_WORLD);
        MPI_Recv(prevBottomSolution.data(), rcount, MPI_DOUBLE, sourceProcess, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        // пересылаем верхние строки всеми процессами кроме нулевого
        MPI_Send(temp.data(), rcount, MPI_DOUBLE, sourceProcess, 46, MPI_COMM_WORLD);
        MPI_Recv(nextTopSolution.data(), scount, MPI_DOUBLE, destProcess, 46, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        // внутренние строки (красные)
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = ((i + 1) % 2 + 1); j < n - 1; j += 2)
                solution[i * n + j] =
                        (temp[(i + 1) * n + j] + temp[(i - 1) * n + j] + temp[i * n + (j + 1)] + temp[i * n + (j - 1)] +
                         pow(h, 2) * right_part((nums_local + i) * h, j * h)) * yMultiplayer;

        // верхние строки (красные)
        if (myId != 0)
            for (int j = 2; j < n - 1; j += 2)
                solution[j] = (temp[n + j] + prevBottomSolution[j] + temp[j + 1] + temp[j - 1] +
                               pow(h, 2) * right_part(nums_local * h, j * h)) * yMultiplayer;

        // нижние строки (красные)
        if (myId != numOfProcessors - 1)
            for (int j = 1 + str_local % 2; j < n - 1; j += 2)
                solution[(str_local - 1) * n + j] =
                        (nextTopSolution[j] + temp[(str_local - 2) * n + j] + temp[(str_local - 1) * n + (j + 1)] +
                         temp[(str_local - 1) * n + (j - 1)] +
                         pow(h, 2) * right_part((nums_local + (str_local - 1)) * h, j * h)) * yMultiplayer;

        //пересылаем нижние строки всеми процессами кроме последнего
        MPI_Send(solution.data() + (str_local - 1) * n, scount, MPI_DOUBLE, destProcess, 42, MPI_COMM_WORLD);
        MPI_Recv(prevBottomSolution.data(), rcount, MPI_DOUBLE, sourceProcess, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        // пересылаем верхние строки всеми процессами кроме нулевого
        MPI_Send(solution.data(), rcount, MPI_DOUBLE, sourceProcess, 46, MPI_COMM_WORLD);
        MPI_Recv(nextTopSolution.data(), scount, MPI_DOUBLE, destProcess, 46, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);


        // внутренние строки (чёрные)
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = (i % 2 + 1); j < n - 1; j += 2)
                solution[i * n + j] =
                        (solution[(i + 1) * n + j] + solution[(i - 1) * n + j] + solution[i * n + (j + 1)] +
                         solution[i * n + (j - 1)] + pow(h, 2) * right_part((nums_local + i) * h, j * h)) *
                        yMultiplayer;

        // верхние строки (чёрные)
        if (myId != 0)
            for (int j = 1; j < n - 1; j += 2)
                solution[j] = (solution[n + j] + prevBottomSolution[j] + solution[j + 1] + solution[j - 1] +
                               pow(h, 2) * right_part(nums_local * h, j * h)) * yMultiplayer;

        // нижние строки (чёрные)
        if (myId != numOfProcessors - 1)
            for (int j = 1 + (str_local - 1) % 2; j < n - 1; j += 2)
                solution[(str_local - 1) * n + j] =
                        (nextTopSolution[j] + solution[(str_local - 2) * n + j] +
                         solution[(str_local - 1) * n + (j + 1)] +
                         solution[(str_local - 1) * n + (j - 1)] +
                         pow(h, 2) * right_part((nums_local + (str_local - 1)) * h, j * h)) * yMultiplayer;

        norm_local = calcDiff(temp, solution);

        MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (norm_err < COMPARE_RATE)
            flag = false;
    }
}

void Helmholtz::calcRedAndBlackSendAndReceive() {
    while (flag) {
        iterationsNum++;
        std::swap(temp, solution);

        // пересылаем нижние и верхние строки
        MPI_Sendrecv(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE, destProcess, 42, prevBottomSolution.data(),
                     rcount,
                     MPI_DOUBLE, sourceProcess, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        MPI_Sendrecv(temp.data(), rcount, MPI_DOUBLE, sourceProcess, 46, nextTopSolution.data(), scount, MPI_DOUBLE,
                     destProcess,
                     46, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        // внутренние строки (красные)
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = ((i + 1) % 2 + 1); j < n - 1; j += 2)
                solution[i * n + j] =
                        (temp[(i + 1) * n + j] + temp[(i - 1) * n + j] + temp[i * n + (j + 1)] + temp[i * n + (j - 1)] +
                         pow(h, 2) * right_part((nums_local + i) * h, j * h)) * yMultiplayer;

        // верхние строки (красные)
        if (myId != 0)
            for (int j = 2; j < n - 1; j += 2)
                solution[j] = (temp[n + j] + prevBottomSolution[j] + temp[j + 1] + temp[j - 1] +
                               pow(h, 2) * right_part(nums_local * h, j * h)) * yMultiplayer;

        // нижние строки (красные)
        if (myId != numOfProcessors - 1)
            for (int j = 1 + str_local % 2; j < n - 1; j += 2)
                solution[(str_local - 1) * n + j] =
                        (nextTopSolution[j] + temp[(str_local - 2) * n + j] + temp[(str_local - 1) * n + (j + 1)] +
                         temp[(str_local - 1) * n + (j - 1)] +
                         pow(h, 2) * right_part((nums_local + (str_local - 1)) * h, j * h)) * yMultiplayer;

        //MPI_Barrier;

        // пересылаем нижние и верхние строки
        MPI_Sendrecv(solution.data() + (str_local - 1) * n, scount, MPI_DOUBLE, destProcess, 42,
                     prevBottomSolution.data(), rcount,
                     MPI_DOUBLE, sourceProcess, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        MPI_Sendrecv(solution.data(), rcount, MPI_DOUBLE, sourceProcess, 46, nextTopSolution.data(), scount, MPI_DOUBLE,
                     destProcess, 46, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);


        // внутренние строки (чёрные)
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = (i % 2 + 1); j < n - 1; j += 2)
                solution[i * n + j] =
                        (solution[(i + 1) * n + j] + solution[(i - 1) * n + j] + solution[i * n + (j + 1)] +
                         solution[i * n + (j - 1)] + pow(h, 2) * right_part((nums_local + i) * h, j * h)) *
                        yMultiplayer;

        // верхние строки (чёрные)
        if (myId != 0)
            for (int j = 1; j < n - 1; j += 2)
                solution[j] = (solution[n + j] + prevBottomSolution[j] + solution[j + 1] + solution[j - 1] +
                               pow(h, 2) * right_part(nums_local * h, j * h)) * yMultiplayer;

        // нижние строки (чёрные)
        if (myId != numOfProcessors - 1)
            for (int j = 1 + (str_local - 1) % 2; j < n - 1; j += 2)
                solution[(str_local - 1) * n + j] =
                        (nextTopSolution[j] + solution[(str_local - 2) * n + j] +
                         solution[(str_local - 1) * n + (j + 1)] +
                         solution[(str_local - 1) * n + (j - 1)] +
                         pow(h, 2) * right_part((nums_local + (str_local - 1)) * h, j * h)) * yMultiplayer;


        norm_local = calcDiff(temp, solution);

        MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (norm_err < COMPARE_RATE)
            flag = false;
    }
}

void Helmholtz::calcRedAndBlackISendIReceive() {
    auto *send_req1 = new MPI_Request[2];
    auto *send_req2 = new MPI_Request[2];
    auto *recv_req1 = new MPI_Request[2];
    auto *recv_req2 = new MPI_Request[2];

    // пересылаем верхние и нижние строки temp
    MPI_Send_init(temp.data(), rcount, MPI_DOUBLE, sourceProcess, 0, MPI_COMM_WORLD, send_req1);
    MPI_Recv_init(prevBottomSolution.data(), rcount, MPI_DOUBLE, sourceProcess, 1, MPI_COMM_WORLD, recv_req1);

    MPI_Send_init(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE, destProcess, 1, MPI_COMM_WORLD, send_req1 + 1);
    MPI_Recv_init(nextTopSolution.data(), scount, MPI_DOUBLE, destProcess, 0, MPI_COMM_WORLD, recv_req1 + 1);

    // пересылаем верхние и нижние строки solution
    MPI_Send_init(solution.data(), rcount, MPI_DOUBLE, sourceProcess, 0, MPI_COMM_WORLD, send_req2);
    MPI_Recv_init(prevBottomSolution.data(), rcount, MPI_DOUBLE, sourceProcess, 1, MPI_COMM_WORLD, recv_req2);

    MPI_Send_init(solution.data() + (str_local - 1) * n, scount, MPI_DOUBLE, destProcess, 1, MPI_COMM_WORLD,
                  send_req2 + 1);
    MPI_Recv_init(nextTopSolution.data(), scount, MPI_DOUBLE, destProcess, 0, MPI_COMM_WORLD, recv_req2 + 1);

    while (flag) {
        iterationsNum++;

        std::swap(temp, solution);

        if (iterationsNum % 2 == 0) {
            MPI_Startall(2, send_req1);
            MPI_Startall(2, recv_req1);
        } else {
            MPI_Startall(2, send_req2);
            MPI_Startall(2, recv_req2);
        }

        // внутренние строки (красные)
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = ((i + 1) % 2 + 1); j < n - 1; j += 2)
                solution[i * n + j] =
                        (temp[(i + 1) * n + j] + temp[(i - 1) * n + j] + temp[i * n + (j + 1)] + temp[i * n + (j - 1)] +
                         pow(h, 2) * right_part((nums_local + i) * h, j * h)) * yMultiplayer;

        if (iterationsNum % 2 == 0) {
            MPI_Waitall(2, send_req1, MPI_STATUSES_IGNORE);
            MPI_Waitall(2, recv_req1, MPI_STATUSES_IGNORE);
        } else {
            MPI_Waitall(2, send_req2, MPI_STATUSES_IGNORE);
            MPI_Waitall(2, recv_req2, MPI_STATUSES_IGNORE);
        }

        // верхние строки (красные)
        if (myId != 0)
            for (int j = 2; j < n - 1; j += 2)
                solution[j] = (temp[n + j] + prevBottomSolution[j] + temp[j + 1] + temp[j - 1] +
                               pow(h, 2) * right_part(nums_local * h, j * h)) * yMultiplayer;

        // нижние строки (красные)
        if (myId != numOfProcessors - 1)
            for (int j = 1 + str_local % 2; j < n - 1; j += 2)
                solution[(str_local - 1) * n + j] =
                        (nextTopSolution[j] + temp[(str_local - 2) * n + j] + temp[(str_local - 1) * n + (j + 1)] +
                         temp[(str_local - 1) * n + (j - 1)] +
                         pow(h, 2) * right_part((nums_local + (str_local - 1)) * h, j * h)) * yMultiplayer;

        if (iterationsNum % 2 == 0) {
            MPI_Startall(2, send_req2);
            MPI_Startall(2, recv_req2);
        } else {
            MPI_Startall(2, send_req1);
            MPI_Startall(2, recv_req1);
        }

        // внутренние строки (чёрные)
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = (i % 2 + 1); j < n - 1; j += 2)
                solution[i * n + j] =
                        (solution[(i + 1) * n + j] + solution[(i - 1) * n + j] + solution[i * n + (j + 1)] +
                         solution[i * n + (j - 1)] + pow(h, 2) * right_part((nums_local + i) * h, j * h)) *
                        yMultiplayer;

        if (iterationsNum % 2 == 0) {
            MPI_Waitall(2, send_req2, MPI_STATUSES_IGNORE);
            MPI_Waitall(2, recv_req2, MPI_STATUSES_IGNORE);
        } else {
            MPI_Waitall(2, send_req1, MPI_STATUSES_IGNORE);
            MPI_Waitall(2, recv_req1, MPI_STATUSES_IGNORE);
        }

        // верхние строки (чёрные)
        if (myId != 0)
            for (int j = 1; j < n - 1; j += 2)
                solution[j] = (solution[n + j] + prevBottomSolution[j] + solution[j + 1] + solution[j - 1] +
                               pow(h, 2) * right_part(nums_local * h, j * h)) * yMultiplayer;

        // нижние строки (чёрные)
        if (myId != numOfProcessors - 1)
            for (int j = 1 + (str_local - 1) % 2; j < n - 1; j += 2)
                solution[(str_local - 1) * n + j] =
                        (nextTopSolution[j] + solution[(str_local - 2) * n + j] +
                         solution[(str_local - 1) * n + (j + 1)] +
                         solution[(str_local - 1) * n + (j - 1)] +
                         pow(h, 2) * right_part((nums_local + (str_local - 1)) * h, j * h)) * yMultiplayer;


        norm_local = calcDiff(temp, solution);

        MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (norm_err < COMPARE_RATE)
            flag = false;
    }
}

double Helmholtz::right_part(double x, double y) {
    return 2.0 * sin(M_PI * y) + pow(k, 2) * (1 - x) * x * sin(M_PI * y) + pow(M_PI, 2) * (1 - x) * x * sin(M_PI * y);
}

double Helmholtz::u_exact(double x, double y) {
    return (1.0 - x) * x * sin(M_PI * y);
}

double Helmholtz::calcDiff(const vector<double> &a, const vector<double> &b) {
    double max = 0.0;
    double tmp = 0.0;

    for (int i = 0; i < a.size(); ++i) {
        tmp = std::abs(a[i] - b[i]);
        if (tmp > max)
            max = tmp;
    }
    return max;
}
