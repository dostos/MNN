#include "OpenCLMultiExecution.hpp"

namespace MNN {

OpenCLMultiExecution::OpenCLMultiExecution(std::vector<std::vector<Execution *>> executions)
    :MultiExecution(executions) {}

ErrorCode OpenCLMultiExecution::onResize(const std::vector<std::vector<Tensor *>> &inputs, const std::vector<std::vector<Tensor *>> &outputs) {
    return NO_ERROR;
}

ErrorCode OpenCLMultiExecution::onExecute(const std::vector<std::vector<Tensor *>> &inputs, const std::vector<std::vector<Tensor *>> &outputs) {
    return NO_ERROR;
}

class OpenCLMultiExecutionCreator : public MultiExecution::Creator {
    virtual MultiExecution* onCreate(std::vector<std::vector<Execution *>> executions) const {
        return new OpenCLMultiExecution(executions);
    }
};

static const auto __opencl_multiexecutioncreator_initializer = []() {
    MultiExecution::insertMultiExecutionCreator(new OpenCLMultiExecutionCreator, MNN_FORWARD_OPENCL);
    return true;
}();
}