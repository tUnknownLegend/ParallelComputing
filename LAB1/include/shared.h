#ifndef INC_LAB_SHARED_H
#define INC_LAB_SHARED_H

#if (_MSC_VER >= 1600)
#define ADD_DOTS "../../../"
#else
#define ADD_DOTS "../"
#endif

// input file for matrix
//#define IN_FILE_MATRIX "../../../inputMatrix.txt"
#define IN_FILE_MATRIX "../../../test12.txt"
//#define IN_FILE_MATRIX "../../../testAS.txt"
// input file for vector
#define IN_FILE_VECTOR "../../../inputVector.txt"
//#define IN_FILE_VECTOR "../../../test12.txt"
// output file for matrix
#define OUT_FILE_MATRIX "../outputMatrix.txt"
// output file for vector
#define OUT_FILE_VECTOR "../../../outputVector.txt"
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

void inputMatrix(std::vector<std::vector<TT>> &matrix);

void outputMatrix(const std::vector<std::vector<TT>> &matrix, const std::string& fileName = OUT_FILE_MATRIX);

void outputMatrix(int amtOfVertices);

void outputVector(int amtOfElements);

void inputVector(std::vector<TT> &vect, const std::string &out = IN_FILE_VECTOR);

void outputVector(const std::vector<TT> &vect, const std::string &out = OUT_FILE_VECTOR);

TT normInfVector(const std::vector<TT> &vect);

TT norm1Vector(const std::vector<TT> &vect);

TT normInfMatrix(const std::vector<std::vector<TT>> &matrix);

TT norm1Matrix(const std::vector<std::vector<TT>> &matrix);

std::vector<TT> MultiplicationMatrixvsVector(const std::vector<std::vector<TT>> &matrix, const std::vector<TT> &vect);

TT normDiffer(const std::vector<std::vector<TT>> &A, const std::vector<TT> &b, const std::vector<TT> &x,
              TT(*normVector)(const std::vector<TT> &));

std::vector<std::vector<TT>> transpoceMatrix(const std::vector<std::vector<TT>> &matrix);

std::vector<std::vector<TT>> identityMatrix(int size, TT digit = 1.0);

void outputOnTheScreenMatrix(const std::vector<std::vector<TT>> &matrix);

void outputOnTheScreenVector(const std::vector<TT> &vector);

std::vector<std::vector<TT>>
matrixOperations(const std::vector<std::vector<TT>> &firstM, const std::vector<std::vector<TT>> &secondM,
                 const char &operation);

void matrixDigit(const TT &digit, std::vector<std::vector<TT>> &secondM, const char &operation);

std::vector<TT> vectorOperation(const std::vector<TT> &firstV, const std::vector<TT> &secondV, const char &operation);

void vectorDigit(const TT &digit, std::vector<TT> &secondV, const char &operation);

std::vector<TT> matrixVectorMultiplication(const std::vector<std::vector<TT>> &firstM,
                                           const std::vector<TT> &secondV);

void LDU(const std::vector<std::vector<TT>> &A, std::vector<std::vector<TT>> &L, std::vector<std::vector<TT>> &D,
         std::vector<std::vector<TT>> &U);

void three_diag_init(std::vector<TT> &a, std::vector<TT> &b, std::vector<TT> &c, std::vector<TT> &d, TT one, TT two,
                     TT three, TT four);

std::vector<std::vector<TT>> inverseMatrix(std::vector<std::vector<TT>> &matrix);

void outputOnTheScreenMatrix(const std::vector<std::vector<TT>> &matrix);

void outputOnTheScreenPairVector(const std::vector<std::pair<TT, TT>> &pair);

std::vector<TT> CalcGaussMethod(std::vector<std::vector<TT>> matr, std::vector<TT> vect);

TT l2NormVec(const std::vector<TT> &vec);

TT l2NormMatr(const std::vector<std::vector<TT>> &matrix);

std::vector<TT> vectorMatrixMultiplication(const std::vector<std::vector<TT>> &firstM, const std::vector<TT> &secondV);

std::vector<TT> vectorRDigit(const TT &digit, std::vector<TT> secondV, const char &operation);

double MeasureFuncExecTime(const std::function<void()> &FuncToMeasure);

#endif //INC_LAB_SHARED_H
