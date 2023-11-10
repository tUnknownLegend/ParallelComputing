#ifndef LAB2_HELMHOLTZ_H
#define LAB2_HELMHOLTZ_H

#include <vector>
#include <algorithm>
#include "shared.h"
#include "matrix.h"

Matrix helmholtzSolve(
        double k,
        double h,
        const std::pair<vector<double>, vector<double>> &grid,
        const std::function<double(double, double)> &f
);


double diffHelmholtz(const Matrix &solution, const std::pair<vector<double>, vector<double>> &grid,
                     const std::function<double(double, double)> &calcPreciseSolution);

#endif //LAB2_HELMHOLTZ_H
