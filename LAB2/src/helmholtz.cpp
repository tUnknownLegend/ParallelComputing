#include <cmath>
#include <iostream>
#include "helmholtz.h"

using std::pair;
using std::vector;

Helmholtz::Helmholtz(const vector<pair<double, double>> &inRegion, const double inH, const double inK) {
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

    for (int i = 0; i < this->data.horizontalLength; ++i) {
        this->data.set(i, 0, 0.0);
    }
    for (int j = 0; j < this->data.verticalLength; ++j) {
        this->data.set(0, j, 0.0);
    }
}

Matrix Helmholtz::helmholtzSolve() {
    const double yMultiplayer = 1.0 / (4.0 + pow(k, 2) * pow(h, 2));
    Matrix previous(this->data);

    do {
        Matrix::swap(this->data, previous);

        for (int j = 1; j < grid.second.size() - 1; ++j) {
            for (int i = isOddNumber(j) ? 1 : 2;
                 i < grid.first.size() - 1; i += 2) {
                forFunc(previous, i, j, yMultiplayer);
            }
        }

        for (int j = 1; j < grid.second.size() - 1; ++j) {
            for (int i = isOddNumber(j) ? 2 : 1;
                 i < grid.first.size() - 1; i += 2) {
                forFunc(previous, i, j, yMultiplayer);
            }
        }
    } while (Matrix::frobeniusNorm(this->data, previous) > COMPARE_RATE);


    return this->data;
}

Matrix Helmholtz::jacobiSolve() {
    const double yMultiplayer = 1.0 / (4.0 + pow(k, 2) * pow(h, 2));
    Matrix previous(this->data);


    do {
        Matrix::swap(this->data, previous);

        for (int j = 1; j < grid.second.size() - 1; ++j) {
            for (int i = 1; i < grid.first.size() - 1; ++i) {
                forFunc(previous, i, j, yMultiplayer);
            }
        }

    } while (Matrix::frobeniusNorm(this->data, previous) > COMPARE_RATE);


    return this->data;
}


void Helmholtz::forFunc(Matrix &previous, const int i, const int j, const double yMultiplayer) {
    previous.set(j, i, yMultiplayer * (pow(h, 2) * rightSideFunction(grid.first[i], grid.second[j]) +
                                       (previous.get(j + 1, i) +
                                        previous.get(j - 1, i) +
                                        previous.get(j, i + 1) +
                                        previous.get(j, i - 1))));
};

double Helmholtz::diffOfSolution() {
    double sum = 0.0;
    for (int i = 0; i < this->data.horizontalLength; ++i) {
        for (int j = 0; j < this->data.verticalLength; ++j) {
            sum += std::pow(this->data.get(i, j) - preciseSolution(grid.first[i], grid.second[j]), 2);
        }
    }
    return sqrt(sum);
}

