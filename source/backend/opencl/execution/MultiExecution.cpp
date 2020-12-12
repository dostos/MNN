#include "backend/opencl/execution/MultiExecution.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include "FusionableExecution.hpp"
#include "core/Execution.hpp"

namespace MNN {
namespace OpenCL {
MultiExecution::MultiExecution(std::vector<std::vector<Execution *>> executions, Backend* backend)
    :MNN::MultiExecution(executions, backend), mBackend(dynamic_cast<OpenCLBackend*>(backend)) {
    // TODO : merge ops & prepare kernel
}

ErrorCode MultiExecution::onPrepare(const MultiExecutionTensors &inputs, const MultiExecutionTensors &outputs) {
    std::vector<const KernelContent*> contents;
    KernelCompiler &compiler = mBackend->getOpenCLRuntime()->KernelCompiler();

    for (int subPipelineIdx = 0; subPipelineIdx < mExecutions.size(); subPipelineIdx++) {
        for (int executionIdx = 0; executionIdx < mExecutions[subPipelineIdx].size(); executionIdx++) {
            auto fusionableExecution = dynamic_cast<FusionableExecution *>(mExecutions[subPipelineIdx][executionIdx]);
            std::string kernelName = fusionableExecution->getKernelName();
            std::string programName = fusionableExecution->getProgramName();

            std::string programSource = mBackend->getOpenCLRuntime()->getProgramSource(programName);

            contents.push_back(compiler.parse(kernelName, programSource));
        }
    }

    // Fuse ops
    mContent = compiler.fuse(contents);

    // Compile kernel
    mKernel = mBackend->getOpenCLRuntime()->buildKernelFromSource(mContent->name, mContent->source, {});
    uint32_t argIdx = 0;

    for (int subPipelineIdx = 0; subPipelineIdx < mExecutions.size(); subPipelineIdx++) {
        for (int executionIdx = 0; executionIdx < mExecutions[subPipelineIdx].size(); executionIdx++) {
            auto fusionableExecution = dynamic_cast<FusionableExecution *>(mExecutions[subPipelineIdx][executionIdx]);
            fusionableExecution->onPrepare(inputs[subPipelineIdx][executionIdx], outputs[subPipelineIdx][executionIdx], &mKernel, argIdx, mGlobalWorkSize);
            MNN_ASSERT(fusionableExecution->getGws().size() == 2);
            for (int i = 0; i < 2; i++) {
                mGlobalWorkSize[i] += fusionableExecution->getGws()[i];
                mLocalWorkSize[i] = std::max(fusionableExecution->getLws()[i], mLocalWorkSize[i]);
            }
        }
    }
    return NO_ERROR;    
}

ErrorCode MultiExecution::onExecute(const MultiExecutionTensors &inputs, const MultiExecutionTensors &outputs) {
    //for (int subPipelineIdx = 0; subPipelineIdx < mExecutions.size(); subPipelineIdx++) {
    //    for (int executionIdx = 0; executionIdx < mExecutions[subPipelineIdx].size(); executionIdx++) {
    //        mExecutions[subPipelineIdx][executionIdx]->onExecute(inputs[subPipelineIdx][executionIdx], inputs[subPipelineIdx][executionIdx]);
    //    }
    //}
    
    auto runtime = mBackend->getOpenCLRuntime();
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, runtime);
    return NO_ERROR;
}   
}

class OpenCLMultiExecutionCreator : public MultiExecution::Creator {
    virtual MultiExecution* onCreate(std::vector<std::vector<Execution *>> executions, Backend* backend) const {
        return new OpenCL::MultiExecution(executions, backend);
    }
};

static const auto __opencl_multiexecutioncreator_initializer = []() {
    MultiExecution::insertMultiExecutionCreator(new OpenCLMultiExecutionCreator, MNN_FORWARD_OPENCL);
    return true;
}();
}