#ifndef LAB2_JACOBI_H
#define LAB2_JACOBI_H

#include <vector>
#include <cmath>

using std::vector;
using std::pair;
using std::pow;

void JacobiSendRecv(int myId,
                    int numOfProcessors);

void JacobiSendAndRecv(int myId,
                       int numOfProcessors);

void JacobiISendIRecv(int myId,
                      int numOfProcessors);

#endif //LAB2_JACOBI_H
