#ifndef OpenCLMultiExecution_hpp
#define OpenCLMultiExecution_hpp

#include "core/MultiExecution.hpp"
#include "backend/opencl/core/runtime/OpenCLRuntime.hpp"

namespace MNN {
namespace OpenCL {
class OpenCLBackend;

class MultiExecution : public MNN::MultiExecution
{
public:
    MultiExecution(std::vector<std::vector<Execution *>> executions, Backend *backend);
    virtual ~MultiExecution() = default;

    virtual ErrorCode onPrepare(const MultiExecutionTensors &inputs, const MultiExecutionTensors &outputs) override;
    virtual ErrorCode onExecute(const MultiExecutionTensors &inputs, const MultiExecutionTensors &outputs) override;

private:
    std::vector<uint32_t> mGlobalWorkSize = {0, 0};
    std::vector<uint32_t> mLocalWorkSize = {0, 0, 0, 0};

    OpenCLBackend *mBackend;
    cl::Kernel mKernel;
    const KernelContent *mContent = nullptr;
};
}
}

#endif