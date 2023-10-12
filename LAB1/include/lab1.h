#ifndef LAB1_LAB1_H
#define LAB1_LAB1_H

#include <vector>
#include <algorithm>
#include "shared.h"

using std::vector;
using std::string;

class Matrix {
private:
    int horizontalLength{};
    int verticalLength{};
    vector<double> data{};
    short bucketSize = 32;
public:
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

    double at(const int i, const int j) {
        return data[i * horizontalLength + j];
    }

    double set(const int i, const int j, const double val) {
        return data[i * horizontalLength + j] = val;
    }

    void lu(int verticalL, int horizontalL, int shift = 0);

    void lu();

    void luParallel(int verticalL, int horizontalL, int shift = 0);

    void luParallel();

    void luBlock();

    void luBlockParallel();

    vector<double> getAllData();

    bool isEqual(Matrix *matrix);
};

#endif //LAB1_LAB1_H
