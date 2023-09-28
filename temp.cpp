#include <iostream>
#include <vector>
#include <omp.h>
//#include <windows.h>

using namespace std;

/*
void fillMatrix(vector<double>& matr, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matr[i] = rand() % 20 + rand() % 10 + 1;
    }
}


int main()
{
    int size = 1000;
    vector<double> A(size * size), B(size * size), C(size * size);

    fillMatrix(A, size, size);
    fillMatrix(B, size, size);

    double t1 = omp_get_wtime();
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                C[i * size + j] = A[i * size + k] * B[k * size + j];
            }
        }
    }

    double t2 = omp_get_wtime();
    std::cout << "seq : " << t2 - t1 << endl;

    t1 = omp_get_wtime();
    omp_set_num_threads(4);
//#pragma omp parallel default(none) shared(A,B,C, size) num_threads(8)
 //{
#pragma omp parallel for  collapse(3) schedule(static, 2) shared(A,B,C,size)
        for (int i = 0; i < size; i++) {
            //cout << " No. " << omp_get_thread_num() << " iter i = " << i << endl;;
            for (int j = 0; j < size; j++) {
                //cout << " No. " << omp_get_thread_num() << " iter j = " << j << endl;;
                for (int k = 0; k < size; k++) {
                    //cout << " No. " << omp_get_thread_num() << " i = " << i << " j = " << j << " k = " << k << endl;;
                    C[i * size + j] = A[i * size + k] * B[k * size + j];
                }
            }
        }
 //}
    t2 = omp_get_wtime();
    cout << "par : " << t2 - t1 << endl;
}
*/

void print(const vector<double> &matr, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << matr[i * cols + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    cout << endl;
}

vector<double> mult(const vector<double> &A, const vector<double> &B, int rowA, int colA, int rowB, int colB) {
    vector<double> res(rowA * colB);

    if (colA == rowB) {
        int n = colA;
        for (int i = 0; i < rowA; i++) {
            for (int j = 0; j < colB; j++) {
                for (int k = 0; k < n; k++) {
                    res[i * colB + j] += A[i * colA + k] * B[k * colB + j];
                }
            }
        }
    } else {
        cout << "Can not do multiplication" << endl;
    }

    return res;
}

void fillMatrix(vector<double> &matr, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matr[i] = rand() % 20 + rand() % 10 + 1;
    }
}

void getMatrixPart(vector<double> &newMatr, int newMatrRows, int newMatrCols, const vector<double> &matr, int matrRows,
                   int matrCols,
                   int rowsFrom, int rowsTo, int colsFrom, int colsTo) {

    newMatr.resize(newMatrRows * newMatrCols);

    int k = 0, q = 0;
    for (int i = rowsFrom; i <= rowsTo; i++) {
        for (int j = colsFrom; j <= colsTo; j++) {
            newMatr[k * newMatrCols + q] = matr[i * matrCols + j];
            q++;
        }

        q = 0;
        k++;
    }

}

//rowsFrom, rowsTo - для matrTo (куда вставляем матрицу matrFrom)
void copyMatrixPart(vector<double> &matrTo, int matrToRows, int matrToCols, const vector<double> &matrFrom,
                    int matrFromRows, int matrFromCols, int rowsFrom, int rowsTo, int colsFrom, int colsTo) {

    int k = 0, q = 0;
    for (int i = rowsFrom; i <= rowsTo; i++) {
        for (int j = colsFrom; j <= colsTo; j++) {
            matrTo[i * matrToCols + j] = matrFrom[k * matrFromCols + q];
            q++;
        }

        q = 0;
        k++;
    }
}


void nonBlockLU3(vector<double> &matr, int rows, int cols) {
    //print(matr, rows, cols);
    int x = min(rows - 1, cols);

    for (int i = 0; i < x; i++) {
        for (int j = i + 1; j < rows; j++) {
            matr[j * cols + i] /= matr[i * cols + i];
        }

        for (int k = i + 1; k < rows; k++) {
            for (int q = i + 1; q < cols; q++) {
                matr[k * cols + q] -= matr[k * cols + i] * matr[i * cols + q];
            }
        }
    }

    //print(matr, rows, cols);
}

void nonBlockLU4(vector<double> &matr, int rows, int cols) {
    //print(matr, rows, cols);
    int x = min(rows - 1, cols);
#pragma omp parallel for schedule(guided, 3) shared(matr) //num_threads(4)
    for (int i = 0; i < x; i++) {
        for (int j = i + 1; j < rows; j++) {
            matr[j * cols + i] /= matr[i * cols + i];
        }

        for (int k = i + 1; k < rows; k++) {
            for (int q = i + 1; q < cols; q++) {
                matr[k * cols + q] -= matr[k * cols + i] * matr[i * cols + q];
            }
        }
    }

    //print(matr, rows, cols);
}

/*
void BlockLU(vector<double>& matr, int rows, int cols, int blockSize) {
    auto MakeL = [](vector<double>& A, int rows, int cols) {
        for (int i = 0; i < rows; i++) {
            for (int j = i + 1; j < cols; j++) {
                A[i * cols + j] = 0;
            }
            A[i * cols + i] = 1;
        }
    };

    vector<double> subMatr;
    // i = 0
    // берем 1:12, 1:3
    getMatrixPart(subMatr, rows, blockSize, matr, rows, cols, 1, rows, 1, blockSize);
    nonBlockLU(subMatr, rows, blockSize);
    // возвращаем эту матрицу в А
    copyMatrixPart(matr, rows, cols, subMatr, rows, blockSize, 1, rows, 1, blockSize);
    // решаем СЛАУ, берем матрицу L
    getMatrixPart(subMatr, blockSize, blockSize, matr, rows, cols, 1, blockSize, 1, blockSize);
    MakeL(subMatr, blockSize, blockSize);
}
*/

void checkLU2(const vector<double> &matr, int rows, int cols) {
    vector<double> L(rows * rows);
    vector<double> U(rows * cols);
    print(matr, rows, cols);

    for (int i = 0; i < rows; i++) {
        if (i > cols) {
            for (int j = 0; j < cols; j++) {
                L[i * rows + j] = matr[i * cols + j];
            }
            for (int j = cols; j < i; j++) {
                L[i * rows + j] = 0;
            }
        } else {
            for (int j = 0; j < i; j++) {
                L[i * rows + j] = matr[i * cols + j];
            }
        }

        L[i * rows + i] = 1;
    }


    cout << "----- L : ------" << endl;
    print(L, rows, rows);

    for (int i = 0; i < rows; i++) {
        for (int j = i; j < cols; j++) {
            U[i * cols + j] = matr[i * cols + j];
        }
    }
    if (rows != cols) for (int k = 0; k < cols; k++) U[(rows - 1) * cols + k] = 0; //заполняем нижнюю строку нулями

    cout << "----- U : ------" << endl;
    print(U, rows, cols);

    cout << "----- L * U : -----" << endl;
    vector<double> res = mult(L, U, rows, rows, rows, cols);
    print(res, rows, cols);
}


void procedureNonBlockLU(vector<double> &matr, int rows, int cols) {
    /*
    cout << "----- Begin -----" << endl;
    print(matr, rows, cols);
    nonBlockLU(matr, rows, cols);
    checkLU2(matr, rows, cols);

    cout << "----- End -----" << endl;
    print(matr, rows, cols);
    */
    nonBlockLU4(matr, rows, cols);
}

//перестановка индексов в умножении (шаг 3), в умножении L32 и U32
void BlockLU3(vector<double> &matr, int n, int blockSize) {
    auto MakeL = [](vector<double> &A, int rows, int cols) {
        for (int i = 0; i < rows; i++) {
            for (int j = i + 1; j < cols; j++) {
                A[i * cols + j] = 0;
            }
            A[i * cols + i] = 1;
        }
    };

    vector<double> L, U, sol(blockSize);
    double diag_elem, b_i, sum;
    int w;

    for (int i = 0; i < n - 1; i += blockSize) { // i < n - 1
        //----- шаг 1 : неблочный LU для A(i : n, i : i + b - 1)
        getMatrixPart(L, n - i, blockSize, matr, n, n, i, n - 1, i, i + blockSize - 1);
        nonBlockLU3(L, n - i, blockSize);
        copyMatrixPart(matr, n, n, L, n - i, blockSize, i, n - 1, i, i + blockSize - 1);

        getMatrixPart(U, blockSize, n - blockSize - i, matr, n, n, i, i + blockSize - 1, i + blockSize,
                      n - 1); // было n - b - i + 1
        //----- шаг 2 : A = L^-1 * A
        MakeL(L, blockSize, blockSize);

        //решаем систему уравнений L22 * Anew(j столбец) = Aold(j столбец)
        for (int col = 0; col < n - blockSize - i; col++) {

            for (int q = 0; q < blockSize; q++) {
                diag_elem = L[q * blockSize + q];

                b_i = U[q * (n - blockSize - i) + col]; // matr[(i + q) * n + col];

                sum = 0;
                for (int k = q - 1; k >= 0; k--) {
                    sum += L[q * blockSize + k] * sol[k];
                }

                sol[q] = (b_i - sum) / diag_elem;
            }

            //A[i:i+b, j] = sol[j]
            for (int z = 0; z < blockSize; z++) {
                U[z * (n - blockSize - i) + col] = sol[z];
            }

        }

        copyMatrixPart(matr, n, n, U, blockSize, n - blockSize - i, i, i + blockSize - 1, i + blockSize, n - 1);

        //----- шаг 3: оставшийся блок :
        double a;
        for (int p = i + blockSize; p < n; p++) {
            for (int g = 0; g < blockSize; g++) {
                a = L[(p - i) * blockSize + g];
                for (int l = i + blockSize; l < n; l++) {
                    matr[p * n + l] -= a * U[g * (n - blockSize - i) + (l - blockSize - i)];
                }
            }
        }
    }
}

void BlockLU4(vector<double> &matr, int n, int blockSize) {
    auto MakeL = [](vector<double> &A, int rows, int cols) {
        for (int i = 0; i < rows; i++) {
            for (int j = i + 1; j < cols; j++) {
                A[i * cols + j] = 0;
            }
            A[i * cols + i] = 1;
        }
    };
    int threads = 8;

    vector<double> L, U, sol(blockSize);
    double diag_elem, b_i, sum;
    int w;

    for (int i = 0; i < n - 1; i += blockSize) { // i < n - 1
        //----- шаг 1 : неблочный LU для A(i : n, i : i + b - 1)
        getMatrixPart(L, n - i, blockSize, matr, n, n, i, n - 1, i, i + blockSize - 1);
        nonBlockLU4(L, n - i, blockSize);
        copyMatrixPart(matr, n, n, L, n - i, blockSize, i, n - 1, i, i + blockSize - 1);

        getMatrixPart(U, blockSize, n - blockSize - i, matr, n, n, i, i + blockSize - 1, i + blockSize,
                      n - 1); // было n - b - i + 1

        //----- шаг 2 : A = L^-1 * A

        MakeL(L, blockSize, blockSize);

//#pragma omp parallel for collapse(3) schedule(dynamic,1) shared(L, U) private(diag_elem, b_i, sum, col, q, k) num_threads(threads)
        //распараллелить действия по столбцам
#pragma omp parallel for shared(L, U) private(diag_elem, b_i, sum) //num_threads(threads)
        for (int col = 0; col < n - blockSize - i; ++col) {
            //cout << "col = " << col << " thr = " << omp_get_thread_num() << endl; cout << endl;
            for (int q = 0; q < blockSize; ++q) {
                diag_elem = L[q * blockSize + q];

                b_i = U[q * (n - blockSize - i) + col]; // matr[(i + q) * n + col];

                sum = 0;
                for (int k = q - 1; k >= 0; --k) {
                    sum += L[q * blockSize + k] * sol[k];
                }

                sol[q] = (b_i - sum) / diag_elem;
            }

            //A[i:i+b, j] = sol[j]
            for (int z = 0; z < blockSize; ++z) {
                U[z * (n - blockSize - i) + col] = sol[z];
            }

        }
        copyMatrixPart(matr, n, n, U, blockSize, n - blockSize - i, i, i + blockSize - 1, i + blockSize, n - 1);

        //----- шаг 3: оставшийся блок :
        /*
#pragma omp parallel default(none) shared(matr, L, U, i, blockSize, n) num_threads(threads)
        {
            double a;
#pragma omp for collapse(3) schedule(dynamic, blockSize) private(a)
            for (int p = i + blockSize; p < n; p++) {
                for (int g = 0; g < blockSize; g++) {
                    a = L[(p - i) * blockSize + g];
                    for (int l = i + blockSize; l < n; l++) {
                        matr[p * n + l] -= a * U[g * (n - blockSize - i) + (l - blockSize - i)];
                    }
                }
            }
        }
        */

        double a;
#pragma omp parallel for collapse(2) shared(L, U, matr, n, blockSize, i) private(a) //num_threads(8)
        for (int p = i + blockSize; p < n; p++) {
            for (int g = 0; g < blockSize; g++) {
                a = L[(p - i) * blockSize + g];
                for (int l = i + blockSize; l < n; l++) {
                    matr[p * n + l] -= a * U[g * (n - blockSize - i) + (l - blockSize - i)];
                }
            }
        }

    }
}
//}

void procedureBlockLU(vector<double> &matr, int n, int blockSize) {
    cout << "----- Begin -----" << endl;
    print(matr, n, n);
    BlockLU4(matr, n, blockSize); //BlockLU2
    checkLU2(matr, n, n);

    cout << "----- End -----" << endl;
    print(matr, n, n);
}

int main() {
    int colA, rowA;
    rowA = 1024 * 4;
    colA = 1024 * 4;
    int blockSize = 1;

    vector<double> A(colA * rowA);
    fillMatrix(A, rowA, colA);
    vector<double> A2(A);
    ////////////////////////////////
    double start_time = omp_get_wtime();

    BlockLU4(A, rowA, blockSize);

    double end_time = omp_get_wtime();
    double partime = end_time - start_time;
    cout << "par time : " << partime << endl;
///////////////
    start_time = omp_get_wtime();

    BlockLU3(A, rowA, blockSize);

    end_time = omp_get_wtime();
    double seqtime = end_time - start_time;
    cout << "seq time : " << seqtime << endl;
///////////////
    cout << "speedup  = " << seqtime / partime << endl;


    // procedureBlockLU(A, rowA, blockSize);


    return 0;
}
//BlockLU(A, rowA, blockSize);
//procedureNonBlockLU(A, rowA, colA);
//procedureBlockLU(A, rowA, blockSize);
//nonBlockLU(A, rowA, blockSize);
