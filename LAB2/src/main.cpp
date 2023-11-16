#include <iostream>
#include <mpi.h>
#include "helmholtz.h"

using std::cout;
using std::cin;
using std::endl;


int main(int argc, char **argv) {
    int myId;
    int numOfProcessors;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcessors);
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);

    const auto helmholtz1 = new Helmholtz(myId, numOfProcessors);
    const auto helmholtz2 = new Helmholtz(myId, numOfProcessors);
    const auto helmholtz3 = new Helmholtz(myId, numOfProcessors);
    const auto helmholtz4 = new Helmholtz(myId, numOfProcessors);
    const auto helmholtz5 = new Helmholtz(myId, numOfProcessors);
    const auto helmholtz6 = new Helmholtz(myId, numOfProcessors);

    helmholtz1->solve(JacobiSendReceive, "JacobiSendReceive");
    helmholtz2->solve(JacobiSendAndReceive, "JacobiSendAndReceive");
    helmholtz3->solve(JacobiISendIReceive, "JacobiISendIReceive");

    helmholtz4->solve(RedAndBlackSendReceive, "RedAndBlackSendReceive");
    helmholtz5->solve(RedAndBlackSendAndReceive, "RedAndBlackSendAndReceive");
    helmholtz6->solve(RedAndBlackISendIReceive, "RedAndBlackISendIReceive");

    MPI_Finalize();
    return 0;
}



