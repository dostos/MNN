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
    OpenCLBackend *mBackend;
    cl::Kernel mKernel;
};
}
}

#endif