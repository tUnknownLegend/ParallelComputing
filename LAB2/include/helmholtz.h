#ifndef LAB2_HELMHOLTZ_H
#define LAB2_HELMHOLTZ_H

#include <vector>
#include <cmath>

using std::vector;
using std::pair;
using std::pow;

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
public:
    Helmholtz(int inMyId,
              int inNumOfProcessors);

    void solve();

    void calcJacobiISendIReceive();

    void calcJacobiSendReceive();

    void calcJacobiSendAndReceive();
};

#endif //LAB2_HELMHOLTZ_H