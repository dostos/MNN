#ifndef FusionableExecution_hpp
#define FusionableExecution_hpp

#include "core/Execution.hpp"
#include "backend/opencl/execution/OpenCLMultiExecution.hpp"

namespace MNN {
namespace OpenCL {

class FusionableExecution : public Execution {
public:
    FusionableExecution(Backend *backend, std::string programName = "", std::string kernelName = "");
    virtual ~FusionableExecution() = default;         
    virtual bool fusionable() const override;

    virtual ErrorCode onPrepare(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, cl::Kernel* kernel, uint32_t& argIdx, std::vector<uint32_t> offset) = 0;

    const std::string &getProgramName() const;
    const std::string &getKernelName() const;

    const std::set<std::string> &getBuildOptions() const;

    const std::vector<uint32_t> &getGws() const;
    const std::vector<uint32_t> &getLws() const;

protected:
    std::string mProgramName;
    std::string mKernelName;
    
    std::set<std::string> mBuildOptions;

    std::vector<uint32_t> mGlobalWorkSize;
    std::vector<uint32_t> mLocalWorkSize;
};
}
}

  #endif