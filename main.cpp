#include <iostream>
#include <vector>
#include "NeuralNet.h"

int main() {
    std::vector<unsigned int> topology{2, 2, 1};
    NeuralNet myNet{ topology };


    for(int epoch=0; epoch < 4; epoch++) {
        std::vector<double> inputVals1{0, 0};
        std::vector<double> inputVals2{0, 1};
        std::vector<double> inputVals3{1, 0};
        std::vector<double> inputVals4{1, 1};
        std::vector<double> targetVals0{0};
        std::vector<double> targetVals1{1};

        myNet.feedForward(inputVals1);
        myNet.backPropagation(targetVals0);
        myNet.feedForward(inputVals2);
        myNet.backPropagation(targetVals1);
        myNet.feedForward(inputVals3);
        myNet.backPropagation(targetVals1);
        myNet.feedForward(inputVals4);
        myNet.backPropagation(targetVals0);
    }


    std::vector<double> resultVals;
    myNet.getResults(resultVals);
    for (const auto &item: resultVals) {
        std::cout << item << " ";
    }

    return 0;
}
