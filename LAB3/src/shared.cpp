#include <fstream>
#include <iostream>
#include <random>
#include <omp.h>
#include "shared.h"

using std::ifstream;
using std::vector;
using std::cerr;
using std::ofstream;
using std::cout;
using std::string;

//  This function generates a random double in [i, j]
double GetRandomDouble(double i, double j) {
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(i, j);
    return dis(gen);
}

// Кубическая норма вектора
TT normInfVector(const vector<TT> &vect) {
    TT norm = std::abs(vect[0]);
    for (size_t i = 1; i < vect.size(); ++i) {
        if (norm < std::abs(vect[i]))
            norm = std::abs(vect[i]);
    }
    return norm;
}

// Октэрическая норма вектора
TT norm1Vector(const vector<TT> &vect) {
    TT norm = 0;
    for (double i: vect) {
        norm += std::abs(i);
    }
    return norm;
}

// Кубическая норма матрицы
TT normInfMatrix(const vector<vector<TT>> &matrix) {
    TT norm = 0;

    for (size_t j = 0; j < matrix.size(); ++j) {
        TT sum = 0;
        for (const auto &i: matrix) {
            sum += std::abs(i[j]);
        }
        if (norm < sum)
            norm = sum;
    }
    return norm;
}

// Октаэдрическая норма матрицы
TT norm1Matrix(const vector<vector<TT>> &matrix) {
    TT norm = 0;

    for (const auto &i: matrix) {
        TT sum = 0;
        for (const auto &j: i) {
            sum += std::abs(j);
        }
        if (norm < sum)
            norm = sum;
    }
    return norm;
}

TT l2NormMatr(const vector<vector<TT>> &matrix) {
    TT sum = 0;
    for (auto &i: matrix) {
        for (auto j: i) {
            sum += pow(j, 2);
        }
    }
    return sqrt(sum);
}

TT l2NormVec(const vector<TT> &vec) {
    TT sum = 0;
    for (double i: vec) {
        sum += pow(i, 2);
    }
    return sqrt(sum);
}

vector<TT> MultiplicationMatrixVector(const vector<vector<TT>> &matrix, const vector<TT> &vect) {
    vector<TT> resVector;
    TT s;
    for (size_t i = 0; i < matrix.size(); ++i) {
        s = 0;
        for (size_t j = 0; j < matrix.size(); ++j) {
            s += matrix[i][j] * vect[j];
        }
        resVector.push_back(s);
    }

    return resVector;
}

// Норма невязки 
TT normDiffer(const vector<vector<TT>> &A, const vector<TT> &b, const vector<TT> &x,
              TT(*normVector)(const vector<TT> &)) {
    vector<TT> differ;
    vector<TT> b1 = MultiplicationMatrixVector(A, x);

    for (size_t i = 0; i < b.size(); ++i) {
        differ.push_back(b[i] - b1[i]);
    }
    return normVector(differ);
}

vector<vector<TT>>
matrixOperations(const vector<vector<TT>> &firstM, const vector<vector<TT>> &secondM, const char &operation) {
    vector<vector<TT>> resMatrix(firstM.size(), vector<TT>(firstM.size(), 0));
    switch (operation) {
        case '*':
            for (size_t i = 0; i < firstM.size(); ++i) {
                for (size_t j = 0; j < secondM.size(); ++j) {
                    for (size_t k = 0; k < firstM.size(); ++k) {
                        resMatrix[i][j] += firstM[i][k] * secondM[k][j];
                    }
                }
            }
            break;
        case '+':
            for (size_t i = 0; i < firstM.size(); ++i) {
                for (size_t j = 0; j < secondM.size(); ++j) {
                    resMatrix[i][j] = firstM[i][j] + secondM[i][j];
                }
            }
            break;
        case '-':
            for (size_t i = 0; i < firstM.size(); ++i) {
                for (size_t j = 0; j < secondM.size(); ++j) {
                    resMatrix[i][j] = firstM[i][j] - secondM[i][j];
                }
            }
            break;
        default:
            std::cerr << "error";
    }
    return resMatrix;
}

void matrixDigit(const TT &digit, vector<vector<TT>> &secondM, const char &operation) {
    switch (operation) {
        case '*':
            for (auto &i: secondM) {
                for (auto &j: i) {
                    j *= digit;
                }
            }
            break;
        case '/':
            for (auto &i: secondM) {
                for (auto &j: i) {
                    j /= digit;
                }
            }
            break;
        case '+':
            for (auto &i: secondM) {
                for (auto &j: i) {
                    j += digit;
                }
            }
            break;
        case '-':
            for (auto &i: secondM) {
                for (auto &j: i) {
                    j -= digit;
                }
            }
            break;
        default:
            std::cerr << "error";
    }
}

vector<TT> matrixVectorMultiplication(const vector<vector<TT>> &firstM, const vector<TT> &secondV) {
    vector<TT> result(secondV.size(), 0);

    for (size_t i = 0; i < secondV.size(); ++i) {
        for (size_t j = 0; j < secondV.size(); ++j) {
            result[i] += firstM[i][j] * secondV[j];
        }
    }
    return result;
}

vector<TT> vectorMatrixMultiplication(const vector<vector<TT>> &firstM, const vector<TT> &secondV) {
    vector<TT> result(secondV.size(), 0);

    for (size_t i = 0; i < secondV.size(); ++i) {
        for (size_t j = 0; j < secondV.size(); ++j) {
            result[j] += firstM[j][i] * secondV[i];
        }
    }
    return result;
}

vector<TT> vectorOperation(const vector<TT> &firstV, const vector<TT> &secondV, const char &operation) {
    vector<TT> result(firstV);
    switch (operation) {
        case '*':
            for (size_t i = 0; i < secondV.size(); ++i) {
                result[i] *= secondV[i];
            }
            break;
        case '/':
            for (size_t i = 0; i < secondV.size(); ++i) {
                result[i] /= secondV[i];
            }
            break;
        case '+':
            for (size_t i = 0; i < secondV.size(); ++i) {
                result[i] += secondV[i];
            }
            break;
        case '-':
            for (size_t i = 0; i < secondV.size(); ++i) {
                result[i] -= secondV[i];
            }
            break;
        default:
            std::cerr << "error";
    }
    return result;
}

vector<TT> vectorRDigit(const TT &digit, vector<TT> secondV, const char &operation) {
    switch (operation) {
        case '*':
            for (auto &i: secondV) {
                i *= digit;
            }
            return secondV;
        case '/':
            for (auto &i: secondV) {
                i /= digit;
            }
            return secondV;
        case '+':
            for (auto &i: secondV) {
                i += digit;
            }
            return secondV;
        case '-':
            for (auto &i: secondV) {
                i -= digit;
            }
            return secondV;
        default:
            std::cerr << "error";
            return {};
    }
}

void vectorDigit(const TT &digit, vector<TT> &secondV, const char &operation) {
    switch (operation) {
        case '*':
            for (auto &i: secondV) {
                i *= digit;
            }
            break;
        case '/':
            for (auto &i: secondV) {
                i /= digit;
            }
            break;
        case '+':
            for (auto &i: secondV) {
                i += digit;
            }
            break;
        case '-':
            for (auto &i: secondV) {
                i -= digit;
            }
            break;
        default:
            std::cerr << "error";
    }
}

double MeasureFuncExecTime(const std::function<void()> &FuncToMeasure) {
    const double startingTime = omp_get_wtime();
    FuncToMeasure();
    const double stopTime = omp_get_wtime();

    return ((stopTime - startingTime));
}

bool isOddNumber(const int n) {
    return n % 2 != 0;
}
