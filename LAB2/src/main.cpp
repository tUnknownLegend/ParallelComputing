#include <iostream>
#include "helmholtz.h"

using std::cout;
using std::cin;
using std::endl;

int main() {
    const auto helmholtz = new Helmholtz({
                                             {0.0, 1.0},
                                             {0.0, 1.0}
                                     }, 0.001, 0.001);

    helmholtz->helmholtzSolve();

    cout << "diff: " << helmholtz->diffOfSolution() << endl;

    delete helmholtz;
    return 0;
}
