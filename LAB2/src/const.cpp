#include "const.h"
#include <mpi.h>

using std::vector;

double right_part(double x, double y) {
    return 2.0 * sin(M_PI * y) + pow(k, 2) * (1 - x) * x * sin(M_PI * y) + pow(M_PI, 2) * (1 - x) * x * sin(M_PI * y);
}

double u_exact(double x, double y) {
    return (1.0 - x) * x * sin(M_PI * y);
}

void str_split(int myid, int np, int &str_local, int &nums_local, vector<int> &str_per_proc, vector<int> &nums_start) {
    str_per_proc.resize(np, n / np);
    nums_start.resize(np, 0);

    for (int i = 0; i < n % np; ++i)
        ++str_per_proc[i];

    for (int i = 1; i < np; ++i)
        nums_start[i] = nums_start[i - 1] + str_per_proc[i - 1];

    MPI_Scatter(str_per_proc.data(), 1, MPI_INT, &str_local, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(nums_start.data(), 1, MPI_INT, &nums_local, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

double proverka(const vector<double> &a, const vector<double> &b) {
    double max = 0.0;
    double tmp = 0.0;

    for (int i = 0; i < a.size(); ++i) {
        tmp = std::abs(a[i] - b[i]);
        if (tmp > max)
            max = tmp;
    }
    return max;
}