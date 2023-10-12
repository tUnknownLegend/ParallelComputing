#include <iostream>
#include "lab1.h"

using std::cout;
using std::cin;
using std::endl;

int main() {
    const auto matrixDefault = new Matrix(4096, 4096, 0.0);
    matrixDefault->fillMatrixWithRandomValues();

    auto matrixBlock = new Matrix(*matrixDefault);

    auto matrixParallel = new Matrix(*matrixDefault);

    auto matrixBlockParallel = new Matrix(*matrixDefault);

    cout << "default LU time exec: " <<
         MeasureFuncExecTime([matrixDefault]() { matrixDefault->lu(); })
         << "\nblock LU time exec: "
         << MeasureFuncExecTime([matrixBlock]() { matrixBlock->luBlock(); }) <<
         "\nLU parallel time exec: "
         << MeasureFuncExecTime([matrixParallel]() { matrixParallel->luParallel(); }) <<
         "\nblock LU parallel time exec: "
         << MeasureFuncExecTime([matrixBlockParallel]() { matrixBlockParallel->luBlockParallel(); })
         <<
         endl;

    cout << "matrixBlock is equal to matrixDefault: " <<
         matrixBlock->isEqual(matrixDefault);
    cout << "\nmatrixParallel is equal to matrixDefault: " <<
         matrixParallel->isEqual(matrixDefault);
    cout << "\nmatrixBlockParallel is equal to matrixDefault: " <<
         matrixBlockParallel->isEqual(matrixDefault) << endl;

    delete matrixDefault;
    delete matrixBlock;
    delete matrixParallel;
    delete matrixBlockParallel;
    return 0;
}
