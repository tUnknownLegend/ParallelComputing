#ifndef LAB2_HELMHOLTZ_H
#define LAB2_HELMHOLTZ_H

#include <vector>
#include <algorithm>
#include "shared.h"
#include "matrix.h"
#include <cmath>

class Helmholtz {
private:
    Matrix data;
    vector<std::pair<double, double>> region;
    double k;
    double h;
    std::pair<vector<double>, vector<double>> grid;

    double preciseSolution(const double x, const double y) {
        return (1 - x) * x * sin(M_PI * y);
    }

    double rightSideFunction(const double x, const double y) const {
        return 2 * sin(M_PI * y) + pow(k, 2) * (1 - x) * x * sin(M_PI * y) + pow(M_PI, 2) * (1 - x) * x * sin(M_PI * y);
    };

    void calcRedAndBlackTreePart(const Matrix &previous, double fMultiplayer, double yMultiplayer,
                                 const std::pair<int, int> &firstIterationOptions);

public:
    Helmholtz(std::vector<std::pair<double, double>> inRegion, double inH, double inK);

    Matrix helmholtzSolve();

    double diffOfSolution();
};

#endif //LAB2_HELMHOLTZ_H
