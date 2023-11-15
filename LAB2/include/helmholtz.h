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

    inline double preciseSolution(const double x, const double y) {
        return (1.0 - x) * x * sin(M_PI * y);
    }

    inline double rightSideFunction(const double x, const double y) const {
        return 2.0 * sin(M_PI * y) + pow(k, 2) * (1 - x) * x * sin(M_PI * y) + pow(M_PI, 2) * (1 - x) * x * sin(M_PI * y);
    };

    inline void forFunc(Matrix &previous, int i, int j, double yMultiplayer);

public:
    Helmholtz(const std::vector<std::pair<double, double>>& inRegion, double inH, double inK);

    Matrix helmholtzSolve();

    Matrix jacobiSolve();

    double diffOfSolution();
};

#endif //LAB2_HELMHOLTZ_H
