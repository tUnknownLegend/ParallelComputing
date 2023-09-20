#include <iostream>
#include <vector>
#include <algorithm>

using std::vector;


class matrix {
private:
    unsigned int horizontalLength{};
    unsigned int verticalLength{};
    vector<double> data{};
public:
    matrix(const unsigned int verticalLength,
           const unsigned int horizontalLength) {
        this->verticalLength = verticalLength;
        this->horizontalLength = horizontalLength;
        data.reserve(verticalLength * horizontalLength);
    }

    double at(const unsigned int i, const unsigned int j) {
        return data[i * horizontalLength + j];
    }

    double set(const unsigned int i, const unsigned int j, const double val) {
        return data[i * horizontalLength + j] = val;
    }

    void LU();
};

void matrix::LU() {
    for (unsigned int i = 0; i < std::min(verticalLength - 1, horizontalLength); ++i) {
        for (unsigned int j = i + 1; j < verticalLength; ++j) {
            this->set(j, i, this->at(j, i) / this->at(i, i));
        }

        if (i < horizontalLength) {
            for (unsigned int j = i + 1; j < verticalLength; ++j) {
                for (unsigned int k = i + 1; j < horizontalLength; ++k) {
                    this->set(j, k,
                              this->at(j, k) - this->at(j, i) * this->at(i, k)
                    );
                }
            }
        }
    }

}