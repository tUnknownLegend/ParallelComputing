#include <cmath>
#include <iostream>
#include "helmholtz.h"

using std::pair;
using std::vector;

Helmholtz::Helmholtz(const vector<pair<double, double>> &inRegion, const double inH, const double inK) {
    this->region = inRegion;

    this->h = inH;
    size.first = (region[0].second - region[0].first) / h + 1;

    this->k = inK;
    size.second = (region[1].second - region[1].first) / k + 1;

    Matrix temp(size.first, size.second, 0.0);
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

        for (int j = 1; j < size.second - 1; ++j) {
            for (int i = isOddNumber(j) ? 1 : 2;
                 i < size.first - 1; i += 2) {
                forFunc(previous, i, j, yMultiplayer);
            }
        }

        for (int j = 1; j < size.second - 1; ++j) {
            for (int i = isOddNumber(j) ? 2 : 1;
                 i < size.first - 1; i += 2) {
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

        for (int j = 1; j < size.second - 1; ++j) {
            for (int i = 1; i < size.first - 1; ++i) {
                forFunc(previous, i, j, yMultiplayer);
            }
        }

    } while (Matrix::frobeniusNorm(this->data, previous) > COMPARE_RATE);


    return this->data;
}


void Helmholtz::forFunc(Matrix &previous, const int i, const int j, const double yMultiplayer) {
    previous.set(j, i, yMultiplayer * (pow(h, 2) * rightSideFunction(i * h, j * h) +
                                       (previous.get(j + 1, i) +
                                        previous.get(j - 1, i) +
                                        previous.get(j, i + 1) +
                                        previous.get(j, i - 1))));
};

double Helmholtz::diffOfSolution() {
    double sum = 0.0;
    for (int i = 0; i < this->data.horizontalLength; ++i) {
        for (int j = 0; j < this->data.verticalLength; ++j) {
            sum += std::pow(this->data.get(i, j) - preciseSolution(i * h, j * h), 2);
        }
    }
    return sqrt(sum);
}

