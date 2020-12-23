#include "backend/opencl/execution/OpenCLMultiExecution.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include "FusionableExecution.hpp"
#include "core/Execution.hpp"
#include "core/Pipeline.hpp"
#include <MNN/Interpreter.hpp>

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
    mKernelContent = compiler.fuse(contents);

    // Compile kernel
    mKernel = mBackend->getOpenCLRuntime()->buildKernelFromSource(mKernelContent->name, mKernelContent->source, buildOptions);
    
    for (int subPipelineIdx = 0; subPipelineIdx < mExecutions.size(); subPipelineIdx++) {
        for (int executionIdx = 0; executionIdx < mExecutions[subPipelineIdx].size(); executionIdx++) {
            auto fusionableExecution = dynamic_cast<FusionableExecution *>(mExecutions[subPipelineIdx][executionIdx]);
            
            fusionableExecution->onPrepare(inputs[subPipelineIdx][executionIdx], outputs[subPipelineIdx][executionIdx], &mKernel, mArgIdx, mOffset);
            MNN_ASSERT(fusionableExecution->getGws().size() == 2);
            auto roundGws = OpenCL::roundGws(fusionableExecution->getGws(), fusionableExecution->getLws());
            // Expand gws in fixed dimension
            mGlobalWorkSize[0] += roundGws[0];
            mGlobalWorkSize[1] = std::max(roundGws[1], mGlobalWorkSize[1]);

            mOffset[0] += roundGws[0];

            for (int i = 0; i < 2; i++) {
                mLocalWorkSize[i] = std::max(fusionableExecution->getLws()[i], mLocalWorkSize[i]);
            }
        }
    }
    
    return NO_ERROR;    
}

ErrorCode OpenCLMultiExecution::onExecute() {
    auto runtime = mBackend->getOpenCLRuntime();
    
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    cl_int error = runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, runtime, &event);

    int costTime = (int)runtime->getCostTime(&event);
    
#else
    cl_int error = runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, runtime);
#endif

    if (error != CL_SUCCESS) {
        MNN_PRINT("MultiExecution : %s execution failed\n num args : %d\n %s\n", mContent->name.c_str(), mArgIdx, mKernelContent->source.c_str());
        return NO_EXECUTION;
    } else 
        return NO_ERROR;
}   


 ErrorCode OpenCLMultiExecution::onExecuteCallback(const TensorCallBackWithInfo &enterCallback, const TensorCallBackWithInfo &exitCallback) {

    auto runtime = mBackend->getOpenCLRuntime();
    cl::Event event;
    enterCallback({}, this);
    cl_int error = runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, runtime, &event);
    int costTime = (int)runtime->getCostTime(&event);
    exitCallback({}, this);
    //MNN_PRINT("MultiExecution : %s gws : (%u, %u)\n", mKernelContent->name.c_str(), mGlobalWorkSize[0], mGlobalWorkSize[1]);

    if (error != CL_SUCCESS) {
        MNN_PRINT("MultiExecution : %s execution failed\n num args : %d\n %s\n", mKernelContent->name.c_str(), mArgIdx, mKernelContent->source.c_str());
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