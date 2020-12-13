#include "backend/opencl/execution/OpenCLMultiExecution.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include "FusionableExecution.hpp"
#include "core/Execution.hpp"

namespace MNN {
namespace OpenCL {
OpenCLMultiExecution::OpenCLMultiExecution(std::vector<std::vector<Execution *>> executions, Backend* backend)
    :MNN::MultiExecution(executions, backend), mBackend(dynamic_cast<OpenCLBackend*>(backend)) {
    // TODO : merge ops & prepare kernel
}

ErrorCode OpenCLMultiExecution::onPrepare(const MultiExecutionTensors &inputs, const MultiExecutionTensors &outputs) {
    std::vector<const KernelContent*> contents;
    KernelCompiler &compiler = mBackend->getOpenCLRuntime()->KernelCompiler();
    std::set<std::string> buildOptions;

    for (int subPipelineIdx = 0; subPipelineIdx < mExecutions.size(); subPipelineIdx++) {
        for (int executionIdx = 0; executionIdx < mExecutions[subPipelineIdx].size(); executionIdx++) {
            auto fusionableExecution = dynamic_cast<FusionableExecution *>(mExecutions[subPipelineIdx][executionIdx]);
            std::string kernelName = fusionableExecution->getKernelName();
            std::string programName = fusionableExecution->getProgramName();

            std::string programSource = mBackend->getOpenCLRuntime()->getProgramSource(programName);

            contents.push_back(compiler.parse(kernelName, programSource));
            buildOptions.insert(fusionableExecution->getBuildOptions().begin(), fusionableExecution->getBuildOptions().end());
        }
    }

    // Fuse ops
    mContent = compiler.fuse(contents);

    // Compile kernel
    mKernel = mBackend->getOpenCLRuntime()->buildKernelFromSource(mContent->name, mContent->source, buildOptions);
    
    for (int subPipelineIdx = 0; subPipelineIdx < mExecutions.size(); subPipelineIdx++) {
        for (int executionIdx = 0; executionIdx < mExecutions[subPipelineIdx].size(); executionIdx++) {
            auto fusionableExecution = dynamic_cast<FusionableExecution *>(mExecutions[subPipelineIdx][executionIdx]);
            fusionableExecution->onPrepare(inputs[subPipelineIdx][executionIdx], outputs[subPipelineIdx][executionIdx], &mKernel, mArgIdx, mGlobalWorkSize);
            MNN_ASSERT(fusionableExecution->getGws().size() == 2);
            for (int i = 0; i < 2; i++) {
                mGlobalWorkSize[i] += fusionableExecution->getGws()[i];
                mLocalWorkSize[i] = std::max(fusionableExecution->getLws()[i], mLocalWorkSize[i]);
            }
        }
    }
    return NO_ERROR;    
}

ErrorCode OpenCLMultiExecution::onExecute(const MultiExecutionTensors &inputs, const MultiExecutionTensors &outputs) {
    auto runtime = mBackend->getOpenCLRuntime();
    
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    cl_int error = runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, runtime, &event);

    int costTime = (int)runtime->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us %s\n", costTime, mContent->name.c_str());
#else
    cl_int error = runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, runtime);
#endif

    if (error != CL_SUCCESS) {
        MNN_PRINT("MultiExecution : %s execution failed\n num args : %d\n %s\n", mContent->name.c_str(), mArgIdx, mContent->source.c_str());
        return NO_EXECUTION;
    } else 
        return NO_ERROR;
    }   
}

class OpenCLMultiExecutionCreator : public MultiExecution::Creator {
    virtual MultiExecution* onCreate(std::vector<std::vector<Execution *>> executions, Backend* backend) const {
        return new OpenCL::OpenCLMultiExecution(executions, backend);
    }
};

static const auto __opencl_multiexecutioncreator_initializer = []() {
    MultiExecution::insertMultiExecutionCreator(new OpenCLMultiExecutionCreator, MNN_FORWARD_OPENCL);
    return true;
}();
}