#include <iostream>
#include "lab1.h"

using std::cout;
using std::cin;
using std::endl;

int main() {
    const auto matrixDefault = new Matrix();
    matrixDefault->inputMatrixFromFile();

    auto matrixBlock = new Matrix();
    matrixBlock->inputMatrixFromFile();

    auto matrixParallel = new Matrix();
    matrixParallel->inputMatrixFromFile();

    auto matrixBlockParallel = new Matrix();
    matrixBlockParallel->inputMatrixFromFile();

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
