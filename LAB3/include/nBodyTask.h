#include <string>
#include <fstream>
#include <cmath>

const int num = 4; // Номер разбиения

const double tau = num * 1e-3; // Шаг по времени
const double timeStep = 1e-1;  // Шаг записи в файлы

const double tEnd = 20.0; // Конечный момент времени

const int Nt = round(tEnd / tau);       // Количество шагов по времени
const int tf = round(timeStep / tau); // Коэффициент пропорциональности шагов

const std::pair<double, double> weightRange{1e+10, 1e+10};
const std::pair<double, double> positionRange{-1.0, 1.0};
const std::pair<double, double> velocityRange{-1.0, 1.0};

const double G = 6.67e-11;

struct Body {
    double weight;
    double position[3];
    double velocity[3];
};

std::ostream &operator<<(std::ostream &str, const Body &b);

inline double vectorNorm(const double *r);
inline double vectorNorm2(const double *r);

inline double raiseToThirdPower(double a);

void acceleration(double *a, int N, const double *r, const Body *data, double G);

void read_file(const std::string &file_name, Body *data, int &N);