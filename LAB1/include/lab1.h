#ifndef LAB1_LAB1_H
#define LAB1_LAB1_H

#include <vector>
#include <algorithm>

using std::vector;
using std::string;

class matrix {
private:
    unsigned int horizontalLength{};
    unsigned int verticalLength{};
    vector<double> data{};
public:
    matrix(unsigned int verticalLength,
           unsigned int horizontalLength);

    void inputMatrixFromFile(const string &fileName);


    double at(const unsigned int i, const unsigned int j) {
        return data[i * horizontalLength + j];
    }

    double set(const unsigned int i, const unsigned int j, const double val) {
        return data[i * horizontalLength + j] = val;
    }

    void LU();
};

#endif //LAB1_LAB1_H
