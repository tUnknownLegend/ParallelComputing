#include <iostream>
#include "helmholtz.h"

using std::cout;
using std::cin;
using std::endl;

int main() {
    const auto solve = new Helmholtz({
                                             {0.0, 1.0},
                                             {0.0, 1.0}
                                     }, 0.001, 0.001);

    solve->helmholtzSolve();

    cout << "diff: " << solve->diffHelmholtz() << endl;

    delete solve;
    return 0;
}
