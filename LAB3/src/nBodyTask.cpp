#include "nBodyTask.h"
#include <iomanip>
#include <cmath>

using std::pair;
using std::ostream;
using std::setprecision;
using std::endl;

std::ostream &operator<<(ostream &str, const Body &b) {
    str << setprecision(10) << b.position[0] << " " << b.position[1] << " " << b.position[2] << endl;

    return str;
}

double vectorNorm(const double *r) {
    return sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
}

double vectorNorm2(const double *r) {
    return r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
}

inline double raiseToThirdPower(const double a) {
    return a * a * a;
}

inline double raiseTo32Power(const double a) {
    return a * sqrt(a);
}

void acceleration(double *a, const int N, const double *r, const Body *data, const double G) {

    double buf[3] = {0.0, 0.0, 0.0};

    for (int k = 0; k < 3; ++k)
        a[k] = 0.0;

    Body bod_j{};

    for (int j = 0; j < N; ++j) {

        bod_j = data[j];

        for (int k = 0; k < 3; ++k)
            buf[k] = bod_j.position[k] - r[k];

        const double mul = bod_j.weight / raiseToThirdPower(std::max(vectorNorm(buf), 1e-6));

        for (int k = 0; k < 3; ++k) {
            buf[k] *= mul;
            a[k] +=  buf[k];
        }
    }

    for (int k = 0; k < 3; ++k)
        a[k] *= G;

}

void read_file(const std::string &file_name, Body *data, int &N) {
    std::ifstream F(file_name);

    F >> N;

    for (int i = 0; i < N; ++i)
        F >> data[i].weight >> data[i].position[0] >> data[i].position[1] >> data[i].position[2] >> data[i].velocity[0]
          >> data[i].velocity[1] >> data[i].velocity[2];

    F.close();
}