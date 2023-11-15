#include <iostream>
#include <mpi.h>
#include "jacobi.h"
#include "redAndBlack.h"

using std::cout;
using std::cin;
using std::endl;


int main(int argc, char **argv) {
    int myId;
    int numOfProcessors;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcessors);
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);

    JacobiSendRecv(myId, numOfProcessors);
    JacobiSendAndRecv(myId, numOfProcessors);
    JacobiISendIRecv(myId, numOfProcessors);

    redAndBlackSendRecv(myId, numOfProcessors);
    redBlackSendAndRecv(myId, numOfProcessors);
    redBlackISendIRecv(myId, numOfProcessors);

    MPI_Finalize();
    return 0;
}



