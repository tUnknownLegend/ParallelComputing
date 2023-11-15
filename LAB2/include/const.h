#ifndef LAB2_CONST_H
#define LAB2_CONST_H

#include <vector>
#include <algorithm>
#include <cmath>

const std::vector<std::pair<double, double>> region = {
        {0.0, 1.0},
        {0.0, 1.0}
};
const double h = 0.1;
const double k = 0.1;

const std::pair<int, int> size = {(region[0].second - region[0].first) / h + 1,
                                  (region[1].second - region[1].first) / k + 1};

const int n = size.first;

const double yMultiplayer = 1.0 / (4.0 + pow(k, 2) * pow(h, 2));

double right_part(double x, double y);

double u_exact(double x, double y);

double proverka(const std::vector<double> &a, const std::vector<double> &b);

void str_split(int myid, int np, int &str_local, int &nums_local, std::vector<int> &str_per_proc,
               std::vector<int> &nums_start);

#endif //LAB2_CONST_H
