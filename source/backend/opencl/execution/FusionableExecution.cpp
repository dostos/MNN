#include "FusionableExecution.hpp"

namespace MNN {
namespace OpenCL {
    FusionableExecution::FusionableExecution(Backend* backend) : Execution(backend) {}

    bool FusionableExecution::fusionable() const {
        return true;
    }

    const std::string &FusionableExecution::getProgramName() const {
        return mProgramName;
    }
    
    const std::string &FusionableExecution::getKernelName() const {
        return mKernelName;
    }
}
}