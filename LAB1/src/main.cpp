#include <iostream>
#include "lab1.h"

int main() {
    auto matrix = new Matrix();
    matrix->inputMatrixFromFile();

    matrix->LU();

    matrix->outputMatrixToFile();

    auto matrixBlock = new Matrix();
    matrixBlock->inputMatrixFromFile();

    matrixBlock->LUblock();

    std::cout << "is equal: " << matrixBlock->isEqual(matrix) << "\n";

    return 0;
}
