#ifndef LAB2_HELMHOLTZ_H
#define LAB2_HELMHOLTZ_H

#include <vector>
#include <cmath>
#include <string>

enum SolutionMethod {
    JacobiSendReceive,
    JacobiSendAndReceive,
    JacobiISendIReceive,
    RedAndBlackSendReceive,
    RedAndBlackSendAndReceive,
    RedAndBlackISendIReceive
};

class Helmholtz {
private:
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

    std::vector<int> str_per_proc;
    std::vector<int> nums_start;
    int str_local;
    int nums_local;
    int myId;
    int numOfProcessors;

    int sourceProcess;
    int destProcess;

    int scount;
    int rcount;

    std::vector<double> solution;
    std::vector<double> nextTopSolution;
    std::vector<double> prevBottomSolution;
    std::vector<double> temp;

    double norm_local;
    double norm_err;

    int iterationsNum;
    bool flag;

    std::vector<double> y;
protected:
    void calcJacobiISendIReceive();

    void calcJacobiSendReceive();

    void calcJacobiSendAndReceive();

    void calcRedAndBlackSendReceive();

    void calcRedAndBlackSendAndReceive();

    void calcRedAndBlackISendIReceive();

    double calcDiff(const std::vector<double> &a, const std::vector<double> &b);

public:
    Helmholtz(int inMyId,
              int inNumOfProcessors);

    void solve(SolutionMethod method, const std::string &name);
};

#endif //LAB2_HELMHOLTZ_H