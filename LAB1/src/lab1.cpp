#include <iostream>
#include <fstream>
#include "shared.h"
#include "lab1.h"

matrix::matrix(const unsigned int verticalLength,
               const unsigned int horizontalLength) {
    this->verticalLength = verticalLength;
    this->horizontalLength = horizontalLength;
    data.reserve(verticalLength * horizontalLength);
}

void matrix::inputMatrixFromFile(const string &fileName = IN_FILE_MATRIX) {
    std::ifstream inFile(fileName);
    if (!inFile.is_open()) {
        std::cerr << "error // input.txt open\n";
        return;
    }

    inFile >> verticalLength;
    horizontalLength = verticalLength;
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

void matrix::LU() {
    for (unsigned int i = 0; i < std::min(verticalLength - 1, horizontalLength); ++i) {
        for (unsigned int j = i + 1; j < verticalLength; ++j) {
            set(j, i, at(j, i) / at(i, i));
        }

        if (i < horizontalLength) {
            for (unsigned int j = i + 1; j < verticalLength; ++j) {
                for (unsigned int k = i + 1; j < horizontalLength; ++k) {
                    set(j, k,
                        at(j, k) - at(j, i) * at(i, k)
                    );
                }
            }
        }
    }

}
