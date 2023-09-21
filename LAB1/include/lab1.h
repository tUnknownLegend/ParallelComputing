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
    short bucketSize{};
public:
    Matrix() = default;

    Matrix(unsigned int verticalLength,
           unsigned int horizontalLength);

    void inputMatrixFromFile(const string &fileName = IN_FILE_MATRIX);
    void outputMatrixToFile(const string &fileName = OUT_FILE_MATRIX);


    double at(const unsigned int i, const unsigned int j) {
        return data[i * horizontalLength + j];
    }

    double set(const unsigned int i, const unsigned int j, const double val) {
        return data[i * horizontalLength + j] = val;
    }

    void LU();
};

#endif //LAB1_LAB1_H
