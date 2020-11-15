#include "OpenCLMultiExecution.hpp"
#include "core/Execution.hpp"

namespace MNN {

OpenCLMultiExecution::OpenCLMultiExecution(std::vector<std::vector<Execution *>> executions)
    :MultiExecution(executions) {
    // TODO : merge ops & prepare kernel
}

ErrorCode OpenCLMultiExecution::onResize(const MultiExecutionTensors &inputs, const MultiExecutionTensors &outputs) {

    return NO_ERROR;
}

ErrorCode OpenCLMultiExecution::onExecute(const MultiExecutionTensors &inputs, const MultiExecutionTensors &outputs) {

    for (int subPipelineIdx = 0; subPipelineIdx < mExecutions.size(); subPipelineIdx++) {
        for (int executionIdx = 0; executionIdx < mExecutions[subPipelineIdx].size(); executionIdx++) {
            mExecutions[subPipelineIdx][executionIdx]->onExecute(inputs[subPipelineIdx][executionIdx], inputs[subPipelineIdx][executionIdx]);
        }
    }
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