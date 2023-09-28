#include <iostream>
#include <fstream>
#include <algorithm>
#include <cassert>
#include "lab1.h"

Matrix::Matrix(const unsigned int verticalLength,
               const unsigned int horizontalLength) {
    this->verticalLength = verticalLength;
    this->horizontalLength = horizontalLength;
    bucketSize = (horizontalLength < bucketSize ? horizontalLength / 2 : bucketSize);
    bucketSize = (2 > bucketSize ? 2 : bucketSize);
    data.reserve(verticalLength * horizontalLength);
}

Matrix::Matrix(const unsigned int verticalLength,
               const unsigned int horizontalLength, const double defaultValue) {
    this->verticalLength = verticalLength;
    this->horizontalLength = horizontalLength;
    bucketSize = (horizontalLength < bucketSize ? horizontalLength / 2 : bucketSize);
    bucketSize = (2 > bucketSize ? 2 : bucketSize);
    vector<double> temp(horizontalLength * verticalLength, 0.0);
    data = std::move(temp);
}

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

void Matrix::LU(const unsigned int verticalL, const unsigned int horizontalL) {
    for (unsigned int i = 0; i < std::min(verticalL - 1, horizontalL); ++i) {
        const double divisionCenterElement = 1. / at(i, i);
        for (unsigned int j = i + 1; j < verticalL; ++j) {
            set(j, i, at(j, i) * divisionCenterElement);
        }

//        if (i < horizontalL) {
        for (unsigned int j = i + 1; j < verticalL; ++j) {
            for (unsigned int k = i + 1; k < std::min(horizontalLength, horizontalL); ++k) {
                set(j, k,
                    at(j, k) - at(j, i) * at(i, k)
                );
            }
        }
//        }
    }
}

void Matrix::LU() {
    LU(this->verticalLength, this->horizontalLength);
}

void Matrix::LUblock() {
    assert(verticalLength == horizontalLength);
    for (unsigned int i = 0; i < verticalLength - 1; i += bucketSize) {
        Matrix tempMatrix(horizontalLength, bucketSize, 0.);

        for (size_t k = i; k < horizontalLength; ++k)
            for (size_t l = i; l < i + bucketSize; ++l)
                tempMatrix.set(k - i, l - i, at(k, l));
        tempMatrix.LU();
        for (size_t k = i; k < horizontalLength; ++k)
            for (size_t l = i; l < i + bucketSize; ++l)
                set(k, l, tempMatrix.at(k - i, l - i));

//        LU(verticalLength - i, bucketSize);

        Matrix L22(bucketSize, bucketSize, 0.0);

        for (unsigned int k = i; k < i + bucketSize; ++k) {
            L22.set(k - i, k - i, 1.0);
            for (unsigned int l = i; l < k; ++l) {
                L22.set(k - i, l - i, at(k, l));
            }
        }


        Matrix L32(verticalLength - bucketSize, bucketSize, 0.0);
        for (unsigned int k = i + bucketSize; k < verticalLength; ++k) {
            for (unsigned int l = i; l < i + bucketSize; ++l) {
                L32.set(k - (i + bucketSize), l - i, at(k, l));
            }
        }


        Matrix U23(bucketSize, (horizontalLength - bucketSize), 0.0);
        for (unsigned int k = i; k < i + bucketSize; ++k) {
            for (unsigned int l = i + bucketSize; l < verticalLength; ++l) {
                U23.set(k - i, l - (i + bucketSize), at(k, l));
            }
        }

        for (unsigned int k = 1; k < bucketSize; ++k) {
            for (unsigned int l = 0; l < horizontalLength - (i + bucketSize); ++l) {
                for (unsigned int m = 0; m < k; ++m) {
                    U23.set(k, l,
                            U23.at(k, l) - L22.at(k, m) * U23.at(m, l)
                    );
                }
            }
        }

        for (unsigned int k = i; k < i + bucketSize; ++k) {
            for (unsigned int l = i + bucketSize; l < verticalLength; ++l)
                set(k, l, U23.at(k - i, l - (i + bucketSize)));
        }

        for (unsigned int k = i + bucketSize; k < verticalLength; ++k) {
            for (unsigned int m = 0; m < bucketSize; ++m) {
                for (unsigned int l = i + bucketSize; l < horizontalLength; ++l)
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
