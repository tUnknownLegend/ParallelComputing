#include <fstream>
#include <iostream>
#include <random>
#include <iomanip>
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

void inputMatrix(vector<vector<TT>> &matrix) {
    ifstream inFile(IN_FILE_MATRIX);
    if (!inFile.is_open()) {
        cerr << "error // input.txt open\n";
        return;
    }

    int amtOfVertices = 0;
    inFile >> amtOfVertices;
    matrix.reserve(amtOfVertices);

    {
        vector<TT> str;
        TT node = 0.0;
        for (int i = 0; i < amtOfVertices; ++i) {

            for (int j = 0; j < amtOfVertices; ++j) {
                inFile >> node;
                str.push_back(node);
            }
            matrix.push_back(str);
            str.clear();
        }
    }
    inFile.close();
}

void inputVector(vector<TT> &vect, const string& out) {
    ifstream inFile(out);
    if (!inFile.is_open()) {
        cerr << "error // input.txt open\n";
        return;
    }

    int amtOfVertices = 0;
    inFile >> amtOfVertices;
    vect.reserve(amtOfVertices);

    {
        vector<TT> str;
        TT node = 0.0;
        for (int j = 0; j < amtOfVertices; ++j) {
            inFile >> node;
            str.push_back(node);
        }
        vect = std::move(str);
    }
    inFile.close();
}

void outputVector(int amtOfElements) {
    ofstream outFile(OUT_FILE_VECTOR);
    if (!outFile.is_open()) {
        cerr << "error // output.txt open\n";
        return;
    }

    outFile << amtOfElements << std::endl;

    {
        const int leftBound = 1;
        const int rightBound = 10;
        for (int j = 0; j < amtOfElements; ++j) {
            outFile << std::setprecision(8) << GetRandomDouble(leftBound, rightBound) << " ";
        }
        outFile << std::endl;
    }
    outFile.close();
}

void outputVector(const vector<TT> &vect, const string& out) {
    ofstream outFile(out);
    if (!outFile.is_open()) {
        cerr << "error // output.txt open\n";
        return;
    }

    outFile << vect.size() << std::endl;

    {
        for (auto &el: vect) {
            outFile << std::setprecision(8) << el << " ";
        }
        outFile << std::endl;
    }
    outFile.close();
}

void outputMatrix(const vector<vector<TT>> &matrix, const string& fileName) {
    ofstream outFile(fileName);
    if (!outFile.is_open()) {
        cerr << "error // output.txt open\n";
        return;
    }

    outFile << matrix.size() << std::endl;

    {
        for (auto &raw: matrix) {
            for (auto &el: raw) {
                outFile << std::setprecision(8) << el << " ";
            }
            outFile << std::endl;
        }
    }
    outFile.close();
}

void outputMatrix(int amtOfVertices) {
    ofstream outFile(OUT_FILE_MATRIX);
    if (!outFile.is_open()) {
        cerr << "error // output.txt open\n";
        return;
    }

    outFile << amtOfVertices << std::endl;

    {
        const int leftBound = 0;
        const int rightBound = 10;
        for (int i = 0; i < amtOfVertices; ++i) {

            for (int j = 0; j < amtOfVertices; ++j) {
                outFile << std::setprecision(8) << GetRandomDouble(leftBound, rightBound) << " ";
            }
            outFile << std::endl;
        }
    }
    outFile.close();
}

// вывод матрицы на экран
void outputOnTheScreenMatrix(const vector<vector<TT>> &matrix) {
    for (const auto& i : matrix) {
        for (const auto& j : i) {
            cout << std::setprecision(8) << std::setw(i.size() * 8) << j << ' ';
        }
        cout << std::endl;
    }
}

// вывод вектора на экран
void outputOnTheScreenVector(const std::vector<TT> &vector) {
    for (const auto &i: vector) {
        cout << std::setprecision(8) << i << ' ';
    }
    cout << std::endl;
}

// вывод вектора на экран
void outputOnTheScreenPairVector(const std::vector<std::pair<TT, TT>> &pair) {
    for (const auto &i: pair) {
        cout << std::setprecision(8) << i.second << '[';
        cout << std::setprecision(8) << i.first << "], ";
    }
    cout << std::endl;
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
        for (const auto & i : matrix) {
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

vector<TT> MultiplicationMatrixvsVector(const vector<vector<TT>> &matrix, const vector<TT> &vect) {
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
    vector<TT> b1 = MultiplicationMatrixvsVector(A, x);

    for (size_t i = 0; i < b.size(); ++i) {
        differ.push_back(b[i] - b1[i]);
    }
    return normVector(differ);
}

vector<vector<TT>> transpoceMatrix(const vector<vector<TT>> &matrix) {
    vector<vector<TT>> resMatrix(matrix);
    for (size_t j = 0; j < matrix.size(); ++j) {
        for (size_t i = 0; i < matrix.size(); ++i) {
            resMatrix[j][i] = matrix[i][j];
        }
    }
    return resMatrix;
}

// Единичная матрица
vector<vector<TT>> identityMatrix(int size, TT digit) {
    vector<vector<TT>> resMatrix(size, vector<TT>(size, 0.0));
    for (int i = 0; i < size; ++i) {
        resMatrix[i][i] = digit;
    }
    return resMatrix;
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

// Разложение матрицы на LDU
void LDU(const vector<vector<TT>> &A, vector<vector<TT>> &L, vector<vector<TT>> &D, vector<vector<TT>> &U) {
    for (size_t i = 0; i < A.size(); i++) {
        for (size_t j = 0; j < A.size(); j++) {
            if (i == j) D[i][j] = A[i][j];
            if (i > j) L[i][j] = A[i][j];
            if (i < j) U[i][j] = A[i][j];
        }
    }
}

vector<TT> CalcGaussMethod(vector<vector<TT>> matr, vector<TT> vect) {
    vector<TT> resultVect(vect.size(), 1.0);

    for (size_t k = 0; k < matr[1].size(); ++k) {
        size_t maxValInd = k;
        for (size_t i = k; i < matr.size(); ++i)
        {
            if (std::abs(matr[i][k]) > std::abs(matr[maxValInd][k]))
                maxValInd = i;
        }

        if (maxValInd != k) {
            std::swap(matr[maxValInd], matr[k]);
            std::swap(vect[maxValInd], vect[k]);
        }
        for (size_t i = k + 1; i < matr.size(); ++i) {
            TT coeffProp = matr[i][k] / matr[k][k];

            for (size_t j = k; j < matr[1].size(); ++j) {
                matr[i][j] -= matr[k][j] * coeffProp;
            }

            vect[i] -= vect[k] * coeffProp;
        }
    }
    for (int i = matr.size() - 1; i >= 0; --i) {
        TT sum = 0.0;
        for (size_t j = i + 1; j < matr.size(); ++j) {
            sum = sum + matr[i][j] * resultVect[j];
        }
        resultVect[i] = (resultVect[i] - sum) / matr[i][i];
    }
    return resultVect;
}

// Обратная матрица
vector<vector<TT>> inverseMatrix(vector<vector<TT>> &matrix) {
    vector<TT> res(matrix.size(), 0.0);
    vector<TT> str;
    vector<vector<TT>> resMatrix;
    vector<vector<TT>> EE;
    EE = identityMatrix(matrix.size());

    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix.size(); ++j) {
            str.push_back(EE[j][i]);
        }
        res = CalcGaussMethod(matrix, str);
        resMatrix.push_back(res);
        str.clear();
    }
    return transpoceMatrix(resMatrix);
}

double MeasureFuncExecTime(const std::function<void()> &FuncToMeasure) {
    unsigned int startingTime = clock();
    FuncToMeasure();
    unsigned int stopTime = clock();
    unsigned int searchTime = stopTime - startingTime;   //  exec time
    //cout << "\nSearch time: " << ((double) searchTime) / CLOCKS_PER_SEC << "\n";

    return (((double) searchTime) / CLOCKS_PER_SEC);
}
