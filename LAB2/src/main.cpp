#include <iostream>
#include "helmholtz.h"

using std::cout;
using std::cin;
using std::endl;

int main() {
    const vector<std::pair<double, double>> region = {
            {0.0, 1.0},
            {0.0, 1.0}
    };
    const double h = 0.01;
    const double k = 0.01;

    //0.0136657
    const auto helmholtz = new Helmholtz(region, h, k);
    helmholtz->helmholtzSolve();
    cout << "helmholtz, diff: " << helmholtz->diffOfSolution() << endl;

    const auto jacobi = new Helmholtz(region, h, k);
    jacobi->jacobiSolve();
    cout << "jacobi, diff: " << jacobi->diffOfSolution() << endl;


    delete helmholtz;
    delete jacobi;
    return 0;
}
