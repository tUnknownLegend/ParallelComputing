#ifndef LAB2_REDANDBLACK_H
#define LAB2_REDANDBLACK_H

#include <vector>
#include <cmath>

using std::vector;
using std::pair;
using std::pow;

void redAndBlackSendRecv(int myId,
                         int numOfProcessors);

void redBlackSendAndRecv(int myId,
                         int numOfProcessors);

void redBlackISendIRecv(int myId,
                        int numOfProcessors);


#endif //LAB2_REDANDBLACK_H
