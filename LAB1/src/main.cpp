#include <iostream>
#include "lab1.h"

using std::cout;
using std::cin;
using std::endl;

int main() {
    const auto matrixDefault = new Matrix(1024, 1024, 0.0);
//    matrixDefault->inputMatrixFromFile();
    matrixDefault->fillMatrixWithRandomValues();

    auto matrixBlock = new Matrix(*matrixDefault);
//    matrixBlock->inputMatrixFromFile();

    auto matrixParallel = new Matrix(*matrixDefault);
//    matrixParallel->inputMatrixFromFile();

    auto matrixBlockParallel = new Matrix(*matrixDefault);
//    matrixBlockParallel->inputMatrixFromFile();

    cout << "default LU time exec: " <<
         MeasureFuncExecTime([matrixDefault]() { matrixDefault->LU(); })
         << "\nblock LU time exec: "
         << MeasureFuncExecTime([matrixBlock]() { matrixBlock->LUblock(); }) <<
         "\nLU parallel time exec: "
         << MeasureFuncExecTime([matrixParallel]() { matrixParallel->LUparallel(); }) <<
         "\nblock LU parallel time exec: "
         << MeasureFuncExecTime([matrixBlockParallel]() { matrixBlockParallel->LUblockParallel(); })
         <<
         endl;

    cout << "matrixBlock is equal to matrixDefault: " <<
         matrixBlock->isEqual(matrixDefault);
    cout << "\nmatrixParallel is equal to matrixDefault: " <<
         matrixParallel->isEqual(matrixDefault);
    cout << "\nmatrixBlockParallel is equal to matrixDefault: " <<
         matrixBlockParallel->isEqual(matrixDefault);

    delete matrixDefault;
    delete matrixBlock;
    delete matrixParallel;
    delete matrixBlockParallel;
    return 0;
}
