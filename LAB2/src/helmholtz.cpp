#include <cmath>
#include <iostream>
#include "helmholtz.h"

using std::pair;
using std::vector;

Helmholtz::Helmholtz(const vector<pair<double, double>> inRegion, const double inH, const double inK) {
    this->region = inRegion;
    this->h = inH;
    const int verticalSize = ((region[0].second - region[0].first) / h);
    vector<double> verticalGrid(verticalSize + 1, 0.0);
    for (int i = 0; i < verticalSize + 1; ++i) {
        verticalGrid[i] = (i * (region[0].second - region[0].first) / verticalSize);
    }

    this->k = inK;
    const int horizontalSize = ((region[1].second - region[1].first) / k);
    vector<double> horizontalGrid(horizontalSize + 1, 0.0);
    for (int j = 0; j < horizontalSize + 1; ++j) {
        horizontalGrid[j] = (j * (region[1].second - region[1].first) / horizontalSize);
    }
    this->grid = {verticalGrid, horizontalGrid};
    Matrix temp(grid.first.size(), grid.second.size(), 0.0);
    this->data = temp;

//    for (int i = 0; i < this->data.horizontalSize(); ++i) {
//        this->data.set(i, 0, 0.0);
//    }
//    for (int j = 0; j < this->data.verticalSize(); ++j) {
//        this->data.set(0, j, 0.0);
//    }
}

Matrix Helmholtz::helmholtzSolve() {
    const double yMultiplayer = 1.0 / (4.0 + pow(k, 2) * pow(h, 2));
    const double fMultiplayer = pow(h, 2) * yMultiplayer;
    Matrix previous(this->data);

    do {
        this->data.swap(previous);
        calcRedAndBlackTreePart(previous, fMultiplayer, yMultiplayer, {1, 2});
        calcRedAndBlackTreePart(previous, fMultiplayer, yMultiplayer, {2, 1});
    } while (Matrix::frobeniusNorm(this->data, previous) > COMPARE_RATE);

    return this->data;
}

void Helmholtz::calcRedAndBlackTreePart(const Matrix &previous, const double fMultiplayer, const double yMultiplayer,
                                        const std::pair<int, int> &firstIterationOptions) {
    for (int j = 1; j < grid.second.size() - 1; ++j) {
        for (int i = isOddNumber(j) ? firstIterationOptions.first : firstIterationOptions.second;
             i < grid.first.size() - 1; i += 2) {
            data.set(j, i, fMultiplayer * rightSideFunction(grid.first[i], grid.second[j]) +
                           yMultiplayer * (previous.get(j + 1, i) +
                                           previous.get(j - 1, i) +
                                           previous.get(j, i + 1) +
                                           previous.get(j, i - 1)));
        }
    }
};

double Helmholtz::diffOfSolution() {
    double maxDiff = 0.0;
    for (int i = 0; i < this->data.horizontalSize(); ++i) {
        for (int j = 0; j < this->data.verticalSize(); ++j) {
            maxDiff = std::max(std::abs(this->data.get(i, j) - preciseSolution(grid.first[i], grid.second[j])),
                               maxDiff);
        }
    }
    return maxDiff;
}

