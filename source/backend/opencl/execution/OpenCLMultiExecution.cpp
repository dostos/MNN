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

            auto gws = fusionableExecution->getGws();
            uint32_t roundGws = ROUND_UP(fusionableExecution->getGws()[0], mBackend->getOpenCLRuntime()->getKernelPreferredWorkGroupSize(mKernel));
            fusionableExecution->setGws({roundGws, gws[1]});
            // Expand gws in fixed dimension
            mGlobalWorkSize[0] += roundGws;
            mGlobalWorkSize[1] = std::max(gws[1], mGlobalWorkSize[1]);

            mOffset[0] += roundGws;
        }
    }

    mLocalWorkSize = tuneLocalWS(mGlobalWorkSize, mBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mKernel));
    
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


std::vector<uint32_t> OpenCLMultiExecution::getGlobalWorkloadSize() const {
    return mGlobalWorkSize;
}

std::vector<uint32_t> OpenCLMultiExecution::getLocalWorkloadSize() const {
    return mLocalWorkSize;
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


std::vector<uint32_t> OpenCLMultiExecution::tuneLocalWS(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize) {
    MNN_ASSERT(gws.size() == 2);
    
    auto& tunedLws = mBackend->getOpenCLRuntime()->tunedLwsMap();
    //std::pair<std::string, std::vector<uint32_t>> info = std::make_pair(mName + "conv2dGeneralLocalWS", gws);
    //if (tunedLws.find(info) != tunedLws.end()) {
    //    //printf("conv2dGeneralLocalWS Found! gws:%d %d lws:%d %d\n", gws[0], gws[1], tunedLws[info][0], tunedLws[info][1]);
    //    return tunedLws[info];
    //}
    
    std::vector<uint32_t> lws(3, 1);
    std::vector<uint32_t> lws_prefer(4, 1);
    int min_cost = INT_MAX;
    while(lws[1] <= gws[1]*2 || lws[1] <= 4) {
        lws[0] = 1;
        while(lws[0] <= gws[0]*2  || lws[0] <= 4) {
            if(lws[0]*lws[1]*lws[2] <= maxWorkGroupSize) {
                cl::Event event;
                std::vector<uint32_t> internalGlobalWS(2, 1);
                for (size_t i = 0; i < gws.size(); ++i) {
                    internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
                }
                cl_int error = mBackend->getOpenCLRuntime()->commandQueue().enqueueNDRangeKernel(
                                mKernel, cl::NullRange,
                                cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]),
                                cl::NDRange(lws[0], lws[1]),
                                nullptr, &event);
                MNN_CHECK_CL_SUCCESS(error);

                int cost_time = (int)mBackend->getOpenCLRuntime()->getCostTime(&event);
                if(cost_time < min_cost) {
                    min_cost = cost_time;
                    lws_prefer[0] = lws[0];
                    lws_prefer[1] = lws[1];
                }
            }
            lws[0] *= 2;
        }
        lws[1] *= 2;
    }

    //if (tunedLws.find(info) == tunedLws.end()) {
    //    //printf("conv2dGeneralLocalWS %d Insert! gws:%d %d, lws:%d %d\n", (int)tunedLws.size(), gws[0], gws[1], lws_prefer[0], lws_prefer[1]);
    //    tunedLws.insert(std::make_pair(info, lws_prefer));
    //}
    
    return lws_prefer;
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