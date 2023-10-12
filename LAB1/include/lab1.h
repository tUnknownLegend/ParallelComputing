#ifndef LAB1_LAB1_H
#define LAB1_LAB1_H

#include <vector>
#include <algorithm>
#include "shared.h"

using std::vector;
using std::string;

class Matrix {
private:
    unsigned int horizontalLength{};
    unsigned int verticalLength{};
    vector<double> data{};
    short bucketSize = 32;
public:
    Matrix() = default;

    Matrix(unsigned int verticalLength,
           unsigned int horizontalLength);

    Matrix(unsigned int verticalLength,
           unsigned int horizontalLength, double defaultValue);

    Matrix(const Matrix &matrix);

    void inputMatrixFromFile(const string &fileName = IN_FILE_MATRIX);

    void outputMatrixToFile(const string &fileName = OUT_FILE_MATRIX);

    void outputMatrixToConsole();

    double at(const unsigned int i, const unsigned int j) {
        return data[i * horizontalLength + j];
    }

    double set(const unsigned int i, const unsigned int j, const double val) {
        return data[i * horizontalLength + j] = val;
    }

    void LU(unsigned int verticalL, unsigned int horizontalL);

    void LU();

    void LUparallel(unsigned int verticalL, unsigned int horizontalL);

    void LUparallel();

    void LUblock();

    vector<double> getAllData();

    bool isEqual(Matrix *matrix);
};

#endif //LAB1_LAB1_H
