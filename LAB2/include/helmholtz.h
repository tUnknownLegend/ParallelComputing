#ifndef LAB2_HELMHOLTZ_H
#define LAB2_HELMHOLTZ_H

#include <vector>
#include <cmath>
#include <string>
#include <functional>

enum JacobiSolutionMethod {
    JacobiSendReceive,
    JacobiSendAndReceive,
    JacobiISendIReceive,
    JacobiNone = -1,
};

enum RedAndBlackSolutionMethod {
    RedAndBlackSendReceive,
    RedAndBlackSendAndReceive,
    RedAndBlackISendIReceive,
    RedAndBlackNone = -1,
};

const std::vector<std::pair<double, double>> region = {
        {0.0, 1.0},
        {0.0, 1.0}
};

const double h = 0.1;
const double k = 0.1;

const std::pair<int, int> sizeOfTask = {(region[0].second - region[0].first) / h + 1,
                                        (region[1].second - region[1].first) / k + 1};

const int N = sizeOfTask.first;
const double multiplayer = (4.0 + pow(k, 2) * pow(h, 2));


class Helmholtz {
private:
    static inline void
    JacobiSendRecv(std::vector<double> &solution, std::vector<double> &tempSolution, std::vector<int> &el_num, int myId,
                   int np, int &shift);

    static inline void
    JacobiSendAndRecv(std::vector<double> &solution, std::vector<double> &tempSolution, std::vector<int> &el_num,
                      int myId,
                      int np, int &shift);

    static inline void
    JacobiISendIRecv(std::vector<double> &solution, std::vector<double> &tempSolution, std::vector<int> &el_num,
                     int myId,
                     int np, int &shift);

    static double
    solveMPI(std::vector<double> &solution, std::vector<double> &tempSolution, std::vector<int> &el_num, int myId,
             int np, int &iterationCount,
             const std::function<void(std::vector<double> &solution, std::vector<double> &tempSolution,
                                      std::vector<int> &el_num, int myId,
                                      int np, int &shift)> &calc);

    static double rightSideFunction(double x, double solution);
public:

    static double
    norm(const std::vector<double> &firstVector, const std::vector<double> &secondVector, int startIndex, int endIndex);

    static void
    gatherSolution(std::vector<int> &numOfElement, std::vector<double> &tempSolution, std::vector<double> &solution,
                   std::vector<int> &displacementOfElement, int np,
                   int myId);

    static void preciseSolution(std::vector<double> &u);

    static double
    Jacobi(std::vector<double> &solution, std::vector<double> &tempSolution, std::vector<int> &el_num, int myId, int np,
           int &iterations,
           JacobiSolutionMethod methodType);

    static double
    redAndBlackMethod(std::vector<double> &solution, std::vector<double> &tempSolution, std::vector<int> el_num,
                      int myId, int np, int &iterationCount,
                      RedAndBlackSolutionMethod methodType);
};

#endif //LAB2_HELMHOLTZ_H