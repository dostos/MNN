#include "FusionableExecution.hpp"

namespace MNN {
namespace OpenCL {
    FusionableExecution::FusionableExecution(Backend* backend, std::string programName, std::string kernelName) 
        : Execution(backend), mProgramName(programName), mKernelName(kernelName) {}

    bool FusionableExecution::fusionable() const {
        return true;
    }

    const std::string &FusionableExecution::getProgramName() const {
        return mProgramName;
    }
    
    const std::string &FusionableExecution::getKernelName() const {
        return mKernelName;
    }

    const std::set<std::string> &FusionableExecution::getBuildOptions() const {
        return mBuildOptions;
    }

    const std::vector<uint32_t> &FusionableExecution::getGws() const {
        return mGlobalWorkSize;
    }
    
    void FusionableExecution::setGws(std::vector<uint32_t> gws) {
        mGlobalWorkSize = gws;
    }

    const std::vector<uint32_t> &FusionableExecution::getLws() const {
        return mLocalWorkSize;
    }
}
}