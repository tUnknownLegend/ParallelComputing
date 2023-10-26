#include <iostream>
#include "lab1.h"

using std::cout;
using std::cin;
using std::endl;

int main() {
    const auto matrixDefault = new Matrix(8192, 8192, 0.0);
    matrixDefault->fillMatrixWithRandomValues();

    auto matrixBlock = new Matrix(*matrixDefault);

    auto matrixParallel = new Matrix(*matrixDefault);

    auto matrixBlockParallel = new Matrix(*matrixDefault);

    const auto defaultLuTime = MeasureFuncExecTime([matrixDefault]() { matrixDefault->lu(); });
    const auto defaultParallelLuTime = MeasureFuncExecTime([matrixBlock]() { matrixBlock->luBlock(); });
    const auto blockLuTime = MeasureFuncExecTime([matrixParallel]() { matrixParallel->luParallel(); });
    const auto blockParallelLuTime = MeasureFuncExecTime(
            [matrixBlockParallel]() { matrixBlockParallel->luBlockParallel(); });

    cout << "default LU time exec: " <<
         defaultLuTime
         << "\nblock LU time exec: "
         << blockLuTime <<
         "\nLU parallel time exec: "
         << defaultParallelLuTime <<
         "\nblock LU parallel time exec: "
         << blockParallelLuTime
         <<
         endl;

    cout << "default LU effectivness: " << defaultLuTime / defaultParallelLuTime <<
         "\nblock LU effectivness: " << blockLuTime / blockParallelLuTime << endl;

    cout << "speed up: " << defaultParallelLuTime / blockParallelLuTime << endl;

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
