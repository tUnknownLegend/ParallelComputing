#include <iostream>
#include "lab1.h"

int main() {
    auto matrix = new Matrix();
    matrix->inputMatrixFromFile();

    matrix->LU();

    matrix->outputMatrixToFile();

    return 0;
}
