#include <iostream>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <omp.h>
#include "lab1.h"

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
    vector<double> temp(horizontalLength * verticalLength, 0.0);
    data = std::move(temp);
}

Matrix::Matrix(const Matrix &matrix) {
    this->bucketSize = matrix.bucketSize;
    this->horizontalLength = matrix.horizontalLength;
    this->verticalLength = matrix.verticalLength;
    this->data = matrix.data;
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
            outFile << at(i, j) << " ";
        }
        outFile << std::endl;
    }

    outFile.close();
}

void Matrix::outputMatrixToConsole() {
    std::cout << verticalLength << std::endl;

    for (int i = 0; i < verticalLength; ++i) {

        for (int j = 0; j < horizontalLength; ++j) {
            std::cout << at(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

void Matrix::fillMatrixWithRandomValues() {
    for (int i = 0; i < horizontalLength * verticalLength; ++i) {
        data[i] = GetRandomDouble(-100, 100);
    }
}

void Matrix::lu(const int verticalL, const int horizontalL, const int shift) {
    for (int i = shift; i < std::min(verticalL - 1, horizontalL + shift); ++i) {
        const double divisionCenterElement = 1. / at(i, i);
        for (int j = i + 1; j < verticalL; ++j) {
            set(j, i,
                at(j, i) * divisionCenterElement);
        }

        for (int j = i + 1; j < verticalL; ++j) {
            for (int k = i + 1; k < std::min(horizontalLength, horizontalL + shift); ++k) {
                set(j, k,
                    at(j, k) -
                    at(j, i) * at(i, k)
                );
            }
        }
    }
}

void Matrix::lu() {
    lu(this->verticalLength, this->horizontalLength);
}

void Matrix::luParallel(const int verticalL, const int horizontalL, const int shift) {
//#pragma omp parallel for default(none) shared(verticalL, horizontalL)
    for (int i = shift; i < std::min(verticalL - 1, horizontalL + shift); ++i) {
        const double divisionCenterElement = 1. / at(i, i);
#pragma omp parallel for default(none) shared(i, verticalL, divisionCenterElement)
        for (int j = i + 1; j < verticalL; ++j) {
            set(j, i,
                at(j, i) * divisionCenterElement);
        }

#pragma omp parallel for default(none) shared(i, verticalL, horizontalL, shift)
        for (int j = i + 1; j < verticalL; ++j) {
            for (int k = i + 1; k < std::min(horizontalLength, horizontalL + shift); ++k) {
                set(j, k,
                    at(j, k) -
                    at(j, i) * at(i, k)
                );
            }
        }
    }
}

void Matrix::luParallel() {
    luParallel(this->verticalLength, this->horizontalLength);
}

void Matrix::luBlock() {
    assert(verticalLength == horizontalLength);

    Matrix L22(bucketSize, bucketSize, 0.0);
    Matrix L32(verticalLength - bucketSize, bucketSize, 0.0);
    Matrix U23(bucketSize, (horizontalLength - bucketSize), 0.0);

    for (int i = 0; i < verticalLength - 1; i += bucketSize) {
        lu(horizontalLength, bucketSize, i);

        for (int k = i; k < i + bucketSize; ++k) {
            L22.set(k - i, k - i, 1.0);
            for (int l = i; l < k; ++l) {
                L22.set(k - i, l - i, at(k, l));
            }
        }

        for (int k = i + bucketSize; k < verticalLength; ++k) {
            for (int l = i; l < i + bucketSize; ++l) {
                L32.set(k - (i + bucketSize), l - i, at(k, l));
            }
        }

        for (int k = i; k < i + bucketSize; ++k) {
            for (int l = i + bucketSize; l < verticalLength; ++l) {
                U23.set(k - i, l - (i + bucketSize), at(k, l));
            }
        }

        for (int k = 1; k < bucketSize; ++k) {
            for (int l = 0; l < horizontalLength - (i + bucketSize); ++l) {
                for (int m = 0; m < k; ++m) {
                    U23.set(k, l,
                            U23.at(k, l) - L22.at(k, m) * U23.at(m, l)
                    );
                }
            }
        }

        for (int k = i; k < i + bucketSize; ++k) {
            for (int l = i + bucketSize; l < verticalLength; ++l)
                set(k, l, U23.at(k - i, l - (i + bucketSize)));
        }

        for (int k = i + bucketSize; k < verticalLength; ++k) {
            for (int m = 0; m < bucketSize; ++m) {
                for (int l = i + bucketSize; l < horizontalLength; ++l)
                    set(k, l,
                        at(k, l)
                        -
                        L32.at(k - (i + bucketSize), m) *
                        U23.at(m, l - (i + bucketSize))
                    );
            }
        }
    }
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
                L22.set(k - i, l - i, at(k, l));
            }
        }

        for (int k = i + bucketSize; k < verticalLength; ++k) {
            for (int l = i; l < i + bucketSize; ++l) {
                L32.set(k - (i + bucketSize), l - i, at(k, l));
            }
        }

        for (int k = i; k < i + bucketSize; ++k) {
            for (int l = i + bucketSize; l < verticalLength; ++l) {
                U23.set(k - i, l - (i + bucketSize), at(k, l));
            }
        }

        for (int k = 1; k < bucketSize; ++k) {
            for (int l = 0; l < horizontalLength - (i + bucketSize); ++l) {
                for (int m = 0; m < k; ++m) {
                    U23.set(k, l,
                            U23.at(k, l) - L22.at(k, m) * U23.at(m, l)
                    );
                }
            }
        }

        for (int k = i; k < i + bucketSize; ++k) {
            for (int l = i + bucketSize; l < verticalLength; ++l)
                set(k, l, U23.at(k - i, l - (i + bucketSize)));
        }

#pragma omp parallel for default(none) shared(i, L32, U23)
        for (int k = i + bucketSize; k < verticalLength; ++k) {
            for (int m = 0; m < bucketSize; ++m) {
                for (int l = i + bucketSize; l < horizontalLength; ++l)
                    set(k, l,
                        at(k, l)
                        -
                        L32.at(k - (i + bucketSize), m) *
                        U23.at(m, l - (i + bucketSize))
                    );
            }
        }
    }
}

vector<double> Matrix::getAllData() {
    return data;
}

bool Matrix::isEqual(Matrix *matrix) {
    return matrix->getAllData() == data;
}
