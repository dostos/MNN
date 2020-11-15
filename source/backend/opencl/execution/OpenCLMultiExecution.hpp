#include "core/MultiExecution.hpp"

namespace MNN {
class OpenCLMultiExecution : public MultiExecution {
public:
    OpenCLMultiExecution(std::vector<std::vector<Execution *>> executions);
    virtual ~OpenCLMultiExecution() = default;

    virtual ErrorCode onResize(const std::vector<std::vector<Tensor *>> &inputs, const std::vector<std::vector<Tensor *>> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<std::vector<Tensor *>> &inputs, const std::vector<std::vector<Tensor *>> &outputs) override;


};
}

