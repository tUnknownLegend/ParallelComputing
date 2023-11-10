#include <iostream>
#include "helmholtz.h"
#include <cmath>

using std::cout;
using std::cin;
using std::endl;
using std::pair;

double u(const double x, const double y) {
    return (1 - x) * x * sin(M_PI * y);
}

int main() {
    vector<pair<double, double>> region = {{0.0, 1.0},
                                           {0.0, 1.0}};

    const double h = 0.1;
    const int verticalSize = ((region[0].second - region[0].first) / h);
    vector<double> verticalGrid(verticalSize, 0.0);
    verticalGrid.resize(verticalSize + 1);
    for (int i = 0; i < verticalSize + 1; ++i) {
        verticalGrid[i] = (i * (region[0].second - region[0].first) / verticalSize);
    }
    cout << endl;

    const double k = 0.1;
    const int horizontalSize = ((region[1].second - region[1].first) / k);
    vector<double> horizontalGrid(horizontalSize, 0.0);
    horizontalGrid.resize(horizontalSize + 1);
    for (int j = 0; j < horizontalSize + 1; ++j) {
        horizontalGrid[j] = (j * (region[1].second - region[1].first) / horizontalSize);
    }
    cout << endl;

    std::function<double(double,double)> f = [k](const double x, const double y) {
        return 2 * sin(M_PI * y) + pow(k, 2) * (1 - x) * x * sin(M_PI * y) + pow(M_PI, 2) * (1 - x) * x * sin(M_PI * y);
    };

    const Matrix solution = helmholtzSolve(k, h, {verticalGrid, horizontalGrid}, f);

    cout << "diff: " << diffHelmholtz(solution, {verticalGrid, horizontalGrid}, u) << endl;

    return 0;
}
