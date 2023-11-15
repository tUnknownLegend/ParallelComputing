#include <iostream>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <omp.h>
#include <complex>
#include "matrix.h"

Matrix::Matrix(const int verticalLength,
               const int horizontalLength) {
    this->verticalLength = verticalLength;
    this->horizontalLength = horizontalLength;
    bucketSize = (horizontalLength < bucketSize ? horizontalLength / 2 : bucketSize);
    bucketSize = (2 > bucketSize ? 2 : bucketSize);
    data.reserve(verticalLength * horizontalLength);
}

Matrix::Matrix(const int verticalLength,
               const int horizontalLength, const double defaultValue) {
    this->verticalLength = verticalLength;
    this->horizontalLength = horizontalLength;
    bucketSize = (horizontalLength < bucketSize ? horizontalLength / 2 : bucketSize);
    bucketSize = (2 > bucketSize ? 2 : bucketSize);
    vector<double> temp(horizontalLength * verticalLength, defaultValue);
    data = std::move(temp);
}

Matrix::Matrix(const Matrix &matrix) {
    this->bucketSize = matrix.bucketSize;
    this->horizontalLength = matrix.horizontalLength;
    this->verticalLength = matrix.verticalLength;
    std::copy(matrix.data.begin(), matrix.data.end(), back_inserter(this->data));
};

void Matrix::inputMatrixFromFile(const string &fileName) {
    std::ifstream inFile(fileName);
    if (!inFile.is_open()) {
        std::cerr << "error // file open\n";
        return;
    }

    inFile >> verticalLength;
    horizontalLength = verticalLength;
    bucketSize = (horizontalLength < bucketSize ? horizontalLength / 2 : bucketSize);
    bucketSize = (2 > bucketSize ? 2 : bucketSize);
    data.reserve(verticalLength * horizontalLength);
    {
        double node = 0.0;
        for (int i = 0; i < verticalLength; ++i) {
            for (int j = 0; j < horizontalLength; ++j) {
                inFile >> node;
                data.push_back(node);
            }
        }
    }
    inFile.close();
}

void Matrix::outputMatrixToFile(const string &fileName) {
    std::ofstream outFile(fileName);
    if (!outFile.is_open()) {
        std::cerr << "error // file open\n";
        return;
    }

    outFile << verticalLength << std::endl;

    for (int i = 0; i < verticalLength; ++i) {

        for (int j = 0; j < horizontalLength; ++j) {
            outFile << get(i, j) << " ";
        }
        outFile << std::endl;
    }

    outFile.close();
}

void Matrix::outputMatrixToConsole() {
    std::cout << verticalLength << std::endl;

    for (int i = 0; i < verticalLength; ++i) {

        for (int j = 0; j < horizontalLength; ++j) {
            std::cout << get(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

void Matrix::fillMatrixWithRandomValues() {
#pragma omp parallel for default(none)
    for (int i = 0; i < horizontalLength * verticalLength; ++i) {
        data[i] = GetRandomDouble(-100, 100);
    }
}

void Matrix::luParallel(const int verticalL, const int horizontalL, const int shift) {
    for (int i = shift; i < std::min(verticalL - 1, horizontalL + shift); ++i) {
        const double divisionCenterElement = 1. / get(i, i);
#pragma omp parallel for default(none) shared(i, verticalL, divisionCenterElement)
        for (int j = i + 1; j < verticalL; ++j) {
            set(j, i,
                get(j, i) * divisionCenterElement);
        }

#pragma omp parallel for default(none) shared(i, verticalL, horizontalL, shift)
        for (int j = i + 1; j < verticalL; ++j) {
            for (int k = i + 1; k < std::min(horizontalLength, horizontalL + shift); ++k) {
                set(j, k,
                    get(j, k) -
                    get(j, i) * get(i, k)
                );
            }
        }
    }
}

void Matrix::luParallel() {
    luParallel(this->verticalLength, this->horizontalLength);
}

void Matrix::luBlockParallel() {
    assert(verticalLength == horizontalLength);

    Matrix L22(bucketSize, bucketSize, 0.0);
    Matrix L32(verticalLength - bucketSize, bucketSize, 0.0);
    Matrix U23(bucketSize, (horizontalLength - bucketSize), 0.0);

    for (int i = 0; i < verticalLength - 1; i += bucketSize) {
        luParallel(horizontalLength, bucketSize, i);

        for (int k = i; k < i + bucketSize; ++k) {
            L22.set(k - i, k - i, 1.0);
            for (int l = i; l < k; ++l) {
                L22.set(k - i, l - i, get(k, l));
            }
        }

        for (int k = i + bucketSize; k < verticalLength; ++k) {
            for (int l = i; l < i + bucketSize; ++l) {
                L32.set(k - (i + bucketSize), l - i, get(k, l));
            }
        }

        for (int k = i; k < i + bucketSize; ++k) {
            for (int l = i + bucketSize; l < verticalLength; ++l) {
                U23.set(k - i, l - (i + bucketSize), get(k, l));
            }
        }

        for (int k = 1; k < bucketSize; ++k) {
            for (int l = 0; l < horizontalLength - (i + bucketSize); ++l) {
                for (int m = 0; m < k; ++m) {
                    U23.set(k, l,
                            U23.get(k, l) - L22.get(k, m) * U23.get(m, l)
                    );
                }
            }
        }

        for (int k = i; k < i + bucketSize; ++k) {
            for (int l = i + bucketSize; l < verticalLength; ++l)
                set(k, l, U23.get(k - i, l - (i + bucketSize)));
        }

#pragma omp parallel for default(none) shared(i, L32, U23)
        for (int k = i + bucketSize; k < verticalLength; ++k) {
            for (int m = 0; m < bucketSize; ++m) {
                for (int l = i + bucketSize; l < horizontalLength; ++l)
                    set(k, l,
                        get(k, l)
                        -
                        L32.get(k - (i + bucketSize), m) *
                        U23.get(m, l - (i + bucketSize))
                    );
            }
        }
    }
}

vector<double> Matrix::getAllData() {
    return data;
}

bool Matrix::isEqual(Matrix *matrix) const {
    return matrix->getAllData() == data;
}


void Matrix::swap(Matrix& firstMatrixToSwap, Matrix& secondMatrixToSwap) {
    std::swap(firstMatrixToSwap.data, secondMatrixToSwap.data);
    std::swap(firstMatrixToSwap.verticalLength, secondMatrixToSwap.verticalLength);
    std::swap(firstMatrixToSwap.horizontalLength, secondMatrixToSwap.horizontalLength);
};

void Matrix::swap(Matrix& secondMatrixToSwap) {
    std::swap(this->data, secondMatrixToSwap.data);
    std::swap(this->verticalLength, secondMatrixToSwap.verticalLength);
    std::swap(this->horizontalLength, secondMatrixToSwap.horizontalLength);
};


double Matrix::frobeniusNorm(const Matrix &left, const Matrix &right) {
    double sum = 0.;
    for (int i = 0; i < left.horizontalLength; ++i) {
        for (int j = 0; j < left.verticalLength; ++j) {
            sum += pow(left.get(i, j) - right.get(i, j), 2);
        }
    }

    return sqrt(sum);
}
