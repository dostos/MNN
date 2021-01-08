#ifndef OpenCLMultiExecution_hpp
#define OpenCLMultiExecution_hpp

#include "core/MultiExecution.hpp"
#include "backend/opencl/core/runtime/OpenCLRuntime.hpp"

namespace MNN {
namespace OpenCL {
class OpenCLBackend;

class OpenCLMultiExecution : public MNN::MultiExecution
{
public:
    OpenCLMultiExecution(std::vector<std::vector<Execution *>> executions, Backend *backend);
    virtual ~OpenCLMultiExecution() = default;

    virtual ErrorCode onPrepare(const MultiExecutionTensors &inputs, const MultiExecutionTensors &outputs) override;
    virtual ErrorCode onExecute() override;
    virtual ErrorCode onExecuteCallback(const TensorCallBackWithInfo &enterCallback, const TensorCallBackWithInfo &exitCallback) override;

    virtual std::vector<uint32_t> getGlobalWorkloadSize() const override;
    virtual std::vector<uint32_t> getLocalWorkloadSize() const override;

private:
    std::vector<uint32_t> tuneLocalWS(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize);
    uint32_t mArgIdx = 0;
    std::vector<uint32_t> mGlobalWorkSize = {0, 0};
    std::vector<uint32_t> mOffset = {0, 0};
    std::vector<uint32_t> mLocalWorkSize = {0, 0, 0, 0};

    OpenCLBackend *mBackend;
    cl::Kernel mKernel;
    const KernelContent *mKernelContent = nullptr;
};
}
}

#endif