#include <iostream>
#include "helmholtz.h"
#include "mpi.h"

using std::cout;
using std::cin;
using std::endl;

int main(int argc, char **argv) {
    int myid, np;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    const vector<std::pair<double, double>> region = {
            {0.0, 1.0},
            {0.0, 1.0}
    };
    const double h = 0.1;
    const double k = 0.1;

    //0.0136657
    const auto helmholtz = new Helmholtz(region, h, k);
    helmholtz->helmholtzSolve();
    cout << "helmholtz, diff: " << helmholtz->diffOfSolution() << endl;

    const auto jacobi = new Helmholtz(region, h, k);
    jacobi->jacobiSolve();
    cout << "jacobi, diff: " << jacobi->diffOfSolution() << endl;

    delete helmholtz;
    delete jacobi;

    MPI_Finalize();
    return 0;
}
