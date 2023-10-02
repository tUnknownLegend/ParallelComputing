#ifndef INC_LAB_SHARED_H
#define INC_LAB_SHARED_H

#if (_MSC_VER >= 1600)
#define ADD_DOTS "../../../"
#else
#define ADD_DOTS "../"
#endif

// input file for matrix
#define IN_FILE_MATRIX ADD_DOTS"inputMatrix.txt"
// input file for vector
#define IN_FILE_VECTOR ADD_DOTS"inputVector.txt"
// output file for matrix
#define OUT_FILE_MATRIX ADD_DOTS"outputMatrix.txt"
// output file for vector
#define OUT_FILE_VECTOR ADD_DOTS"outputVector.txt"
// compare for double
#define COMPARE_RATE 10e-6
// zero division error
#define DIVISTION_ERROR 10e-7

#include <vector>
#include <algorithm>
#include <string>

#define TT double

//  This function generates a random TT in [i, j]
double GetRandomDouble(double i, double j);

TT normInfVector(const std::vector<TT> &vect);

TT norm1Vector(const std::vector<TT> &vect);

TT normInfMatrix(const std::vector<std::vector<TT>> &matrix);

TT norm1Matrix(const std::vector<std::vector<TT>> &matrix);

std::vector<TT> MultiplicationMatrixvsVector(const std::vector<std::vector<TT>> &matrix, const std::vector<TT> &vect);

TT normDiffer(const std::vector<std::vector<TT>> &A, const std::vector<TT> &b, const std::vector<TT> &x,
              TT(*normVector)(const std::vector<TT> &));

std::vector<std::vector<TT>>
matrixOperations(const std::vector<std::vector<TT>> &firstM, const std::vector<std::vector<TT>> &secondM,
                 const char &operation);

void matrixDigit(const TT &digit, std::vector<std::vector<TT>> &secondM, const char &operation);

std::vector<TT> vectorOperation(const std::vector<TT> &firstV, const std::vector<TT> &secondV, const char &operation);

void vectorDigit(const TT &digit, std::vector<TT> &secondV, const char &operation);

std::vector<TT> matrixVectorMultiplication(const std::vector<std::vector<TT>> &firstM,
                                           const std::vector<TT> &secondV);

TT l2NormVec(const std::vector<TT> &vec);

TT l2NormMatr(const std::vector<std::vector<TT>> &matrix);

std::vector<TT> vectorMatrixMultiplication(const std::vector<std::vector<TT>> &firstM, const std::vector<TT> &secondV);

std::vector<TT> vectorRDigit(const TT &digit, std::vector<TT> secondV, const char &operation);

double MeasureFuncExecTime(const std::function<void()> &FuncToMeasure);

#endif //INC_LAB_SHARED_H
