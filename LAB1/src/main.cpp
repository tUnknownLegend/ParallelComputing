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

    cout << "default LU time exec: " <<
         MeasureFuncExecTime([matrixDefault]() { matrixDefault->LUparallel(); })
         << "\nblock LU time exec: "
         << MeasureFuncExecTime([matrixBlock]() { matrixBlock->LUblock(); }) <<
         endl;

    cout << "is equal: " << matrixBlock->isEqual(matrixDefault);


    delete matrixDefault;
    delete matrixBlock;
    return 0;
}
