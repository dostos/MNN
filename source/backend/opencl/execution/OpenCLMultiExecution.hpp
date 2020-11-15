#include "core/MultiExecution.hpp"

namespace MNN {
class OpenCLMultiExecution : public MultiExecution {
public:
    OpenCLMultiExecution(std::vector<std::vector<Execution *>> executions);
    virtual ~OpenCLMultiExecution() = default;

    virtual ErrorCode onResize(const MultiExecutionTensors &inputs, const MultiExecutionTensors &outputs) override;
    virtual ErrorCode onExecute(const MultiExecutionTensors &inputs, const MultiExecutionTensors &outputs) override;


};
}

