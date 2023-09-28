#include <iostream>
#include <fstream>
#include <algorithm>
#include "lab1.h"

Matrix::Matrix(const unsigned int verticalLength,
               const unsigned int horizontalLength) {
    this->verticalLength = verticalLength;
    this->horizontalLength = horizontalLength;
    bucketSize = (horizontalLength < bucketSize ? horizontalLength / 2 : bucketSize);
    bucketSize = (2 > bucketSize ? 2 : bucketSize);
    data.reserve(verticalLength * horizontalLength);
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

void Matrix::LU(const unsigned int verticalL, const unsigned int horizontalL) {
    for (unsigned int i = 0; i < std::min(verticalL - 1, horizontalL); ++i) {
        const double divisionCenterElement = 1. / at(i, i);
        for (unsigned int j = i + 1; j < verticalL; ++j) {
            set(j, i, at(j, i) * divisionCenterElement);
        }

        if (i < horizontalL) {
            for (unsigned int j = i + 1; j < verticalL; ++j) {
                for (unsigned int k = i + 1; k < std::min(horizontalLength, horizontalL); ++k) {
                    set(j, k,
                        at(j, k) - at(j, i) * at(i, k)
                    );
                }
            }
        }
    }
}

void Matrix::LU() {
    LU(this->verticalLength, this->horizontalLength);
}

void Matrix::LUblock() {
    for (unsigned int i = 0; i < verticalLength - 1; ++i) {
        LU(verticalLength - 1, bucketSize);

//        vector<double> L22(
        vector<double> L22(bucketSize * bucketSize, 0.0);
//                data.begin() + i * (horizontalLength + 1),
//                data.begin() + (i + bucketSize - 1) * (horizontalLength + 1));

        for (unsigned int k = i; k < i + bucketSize; ++k) {
            L22[(k - i) * verticalLength + k - i] = 1.;
            for (unsigned int l = i; l < k; ++l) {
                L22[(k - i) * verticalLength + l - i] = at(k, l);
            }
        }

//        for (unsigned int k = i; k < i + bucketSize; ++k) {
//            L22[(k - i) * bucketSize + (k - i)] = 1.0;
//        }

//        vector<double> L32(
//                data.begin() + (i + bucketSize) * horizontalLength + i,
//                data.begin() + (horizontalLength - 1) * (horizontalLength) + (i + bucketSize - 1));

//        vector<double> U23(
//                data.begin() + (i) * horizontalLength + i + bucketSize,
//                data.begin() + (i + bucketSize - 1) * (horizontalLength) + horizontalLength - 1);
        vector<double> L32((verticalLength - bucketSize) * bucketSize, 0.0);
        for (unsigned int k = i + bucketSize; k < verticalLength; ++k) {
            for (unsigned int l = i; l < i + bucketSize; ++l) {
                L32[(k - (i + bucketSize)) * verticalLength + l - i] = at(k, l);
            }
        }

        vector<double> U23(bucketSize * (horizontalLength - bucketSize), 0.0);
        for (unsigned int k = i; k < i + bucketSize; ++k) {
            for (unsigned int l = i + bucketSize; l < verticalLength; ++l) {
                U23[(k - i) * verticalLength + l - (i + bucketSize)] = at(k, l);
            }
        }

        for (unsigned int k = 1; k < bucketSize; ++k) {
            for (unsigned int l = 0; l < verticalLength - (i + bucketSize); ++l) {
                for (unsigned int m = 0; m < k; ++m) {
                    U23[k * verticalLength + l] -=
                            L22[k * verticalLength + m] * U23[m * verticalLength + l];
                }
            }
        }
        for (unsigned int k = i; k < i + bucketSize; ++k) {
            for (unsigned int l = i + bucketSize; l < verticalLength; ++l)
                set(k, l, U23[(k - i) * verticalLength + l - (i + bucketSize)]);
        }

        for (size_t k = i + bucketSize; k < verticalLength; ++k) {
            for (size_t m = 0; m < bucketSize; ++m) {
                for (size_t l = i + bucketSize; l < horizontalLength; ++l)
                    set(k, l,
                        at(k, l) -
                        L32[(k - (i + bucketSize)) * verticalLength + m] *
                        U23[m * verticalLength + l - (i + bucketSize)]);
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
