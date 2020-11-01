#ifndef ConvExpr_hpp
#define ConvExpr_hpp

#include <MNN/expr/Expr.hpp>

MNN::Express::VARP ConvExpr(
    std::vector<int> inputShade,
    int outputChannel, 
    std::vector<int> kernelSize, 
    std::vector<int> stride, 
    std::vector<int> dilate, 
    std::vector<int> pads,
    int group = 1);

#endif // ConvExpr_hpp
