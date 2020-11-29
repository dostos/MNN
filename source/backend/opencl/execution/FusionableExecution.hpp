#ifndef FusionableExecution_hpp
#define FusionableExecution_hpp

#include "core/Execution.hpp"
#include "backend/opencl/execution/MultiExecution.hpp"

namespace MNN {
namespace OpenCL {

class FusionableExecution : public Execution {
public:
    FusionableExecution(Backend *backend);
    virtual ~FusionableExecution() = default;         
    virtual bool fusionable() const override;

    const std::string &getProgramName() const;
    const std::string &getKernelName() const;

protected:
    std::string mName;
    std::string mProgramName;
    std::string mKernelName;
};
}
}

  #endif