#include <iostream>
#include <fstream>
#include "lab1.h"

Matrix::Matrix(const unsigned int verticalLength,
               const unsigned int horizontalLength) {
    this->verticalLength = verticalLength;
    this->horizontalLength = horizontalLength;
    data.reserve(verticalLength * horizontalLength);
}

void Matrix::inputMatrixFromFile(const string &fileName) {
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

void Matrix::outputMatrixToFile(const string &fileName) {
    std::ofstream outFile(fileName);
    if (!outFile.is_open()) {
        std::cerr << "error // output.txt open\n";
        return;
    }

    outFile << verticalLength << std::endl;

    {
        for (int i = 0; i < verticalLength; ++i) {

            for (int j = 0; j < horizontalLength; ++j) {
                outFile << at(i, j) << " ";
            }
            outFile << std::endl;
        }
    }
    outFile.close();
}

void Matrix::LU() {
    for (unsigned int i = 0; i < std::min(verticalLength - 1, horizontalLength); ++i) {
        for (unsigned int j = i + 1; j < verticalLength; ++j) {
            set(j, i, at(j, i) / at(i, i));
        }

        if (i < horizontalLength) {
            for (unsigned int j = i + 1; j < verticalLength; ++j) {
                for (unsigned int k = i + 1; k < horizontalLength; ++k) {
                    set(j, k,
                        at(j, k) - at(j, i) * at(i, k)
                    );
                }
            }
        }
    }

}
