#include "ConvExpr.hpp"
#include <MNN/expr/ExprCreator.hpp>

using namespace MNN::Express;

VARP ConvExpr(
    std::vector<int> inputShade,
    int outputChannel, 
    std::vector<int> kernelSize, 
    std::vector<int> stride, 
    std::vector<int> dilate, 
    std::vector<int> pads,
    int group) {
    std::vector<int> channel{inputShade[1], outputChannel};

    int numWeights = channel[1] * (channel[0] / group) * kernelSize[0] * kernelSize[1];
    int numBias = channel[1];

    auto x = _Input(inputShade, NC4HW4);

    std::vector<float> weight, bias;
    weight.reserve(numWeights);
    bias.reserve(numBias);

    for (int i = 0; i < numWeights; i++) {
        weight.push_back(i);
    }

    for (int i = 0; i < numBias; i++) {
        bias.push_back(i);
    }

    x = _Conv(std::move(weight), std::move(bias), x, channel, kernelSize, SAME, stride, dilate, group, pads);
   
    return x;
}
