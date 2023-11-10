#include <cmath>
#include <iostream>
#include "helmholtz.h"

Matrix helmholtzSolve(
        const double k,
        const double h,
        const std::pair<vector<double>, vector<double>> &grid,
        const std::function<double(double, double)> &f
) {
    Matrix result(grid.first.size(), grid.second.size(), 0.0);

    const double yMultiplayer = 1.0 / (4.0 + pow(k, 2) * pow(h, 2));
    const double fMultiplayer = pow(h, 2) * yMultiplayer;
    Matrix previous(result);

    auto calcRedAndBlackTreePart = [&result, &previous, &fMultiplayer, &f, &yMultiplayer, &grid](
            const std::pair<int, int> &firstIterationOptions) mutable {
        for (int j = 1; j < grid.second.size() - 1; ++j) {
            for (int i = isOddNumber(j) ? firstIterationOptions.first : firstIterationOptions.second;
                 i < grid.first.size() - 1; i += 2) {
                result.set(j, i, fMultiplayer * f(grid.first[i], grid.second[j]) +
                                 yMultiplayer * (previous.get(j + 1, i) +
                                                 previous.get(j - 1, i) +
                                                 previous.get(j, i + 1) +
                                                 previous.get(j, i - 1)));
            }
        }
    };

    do {
        result.swap(previous);
        calcRedAndBlackTreePart({1, 2});
        calcRedAndBlackTreePart({2, 1});
    } while (Matrix::frobeniusNorm(result, previous) > COMPARE_RATE);

    return result;
}

double diffHelmholtz(const Matrix &solution, const std::pair<vector<double>, vector<double>> &grid,
                     const std::function<double(double, double)> &calcPreciseSolution) {
    double maxDiff = 0.0;
    for (int i = 0; i < solution.horizontalSize(); ++i) {
        for (int j = 0; j < solution.verticalSize(); ++j) {
            std::cout << "solution[" << i << ", " << j << "] = " << solution.get(i, j) << "; " << "PreciseSolution["
                      << i << ", " << j << "] = " << calcPreciseSolution(grid.first[i], grid.second[j]) << "\n";
            maxDiff = std::max(std::abs(solution.get(i, j) - calcPreciseSolution(grid.first[i], grid.second[j])),
                               maxDiff);
        }
    }
    return maxDiff;
}

