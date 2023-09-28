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

    matrixBlock->outputMatrixToConsole();

    std::cout << "\nis equal: " << matrixBlock->isEqual(matrix) << "\n";

    return 0;
}
