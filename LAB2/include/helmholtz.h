#ifndef LAB2_HELMHOLTZ_H
#define LAB2_HELMHOLTZ_H

#include <vector>
#include <cmath>
#include <string>

using std::vector;
using std::pair;
using std::pow;

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
    vector<int> str_per_proc;
    vector<int> nums_start;
    int str_local;
    int nums_local;
    int myId;
    int numOfProcessors;

    int sourceProcess;
    int destProcess;

    int scount;
    int rcount;

    vector<double> solution;
    vector<double> nextTopSolution;
    vector<double> prevBottomSolution;
    vector<double> temp;

    double norm_local;
    double norm_err;

    int iterationsNum;
    bool flag;

    vector<double> y;

protected:
    void calcJacobiISendIReceive();

    void calcJacobiSendReceive();

    void calcJacobiSendAndReceive();

    void calcRedAndBlackSendReceive();

    void calcRedAndBlackSendAndReceive();

    void calcRedAndBlackISendIReceive();

public:
    Helmholtz(int inMyId,
              int inNumOfProcessors);

    void solve(SolutionMethod method, const std::string& name);
};

#endif //LAB2_HELMHOLTZ_H