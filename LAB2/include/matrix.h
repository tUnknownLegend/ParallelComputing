#ifndef LAB2_MATRIX_H
#define LAB2_MATRIX_H

#include <vector>
#include <algorithm>
#include "shared.h"

using std::vector;
using std::string;

class Matrix {
private:
    int horizontalLength{};
    int verticalLength{};
    short bucketSize = 32;
public:
    vector<double> data{};
    Matrix() = default;

    Matrix(int verticalLength,
           int horizontalLength);

    Matrix(int verticalLength,
           int horizontalLength, double defaultValue);

    Matrix(const Matrix &matrix);

    void inputMatrixFromFile(const string &fileName = IN_FILE_MATRIX);

    void outputMatrixToFile(const string &fileName = OUT_FILE_MATRIX);

    void outputMatrixToConsole();

    void fillMatrixWithRandomValues();

    int verticalSize() const;

    int horizontalSize() const;

    double get(const int i, const int j) const {
        return data[i * horizontalLength + j];
    }

    double set(const int i, const int j, const double val) {
        return data[i * horizontalLength + j] = val;
    }

    void luParallel(int verticalL, int horizontalL, int shift = 0);

    void luParallel();

    void luBlockParallel();

    vector<double> getAllData();

    bool isEqual(Matrix *matrix);

    void swap(Matrix& matrixToSwap);

    static double frobeniusNorm(const Matrix &lhs, const Matrix &rhs);
};

#endif // LAB2_MATRIX_H
