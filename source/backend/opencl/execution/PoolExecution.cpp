//
//  PoolExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/PoolExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

PoolExecution::PoolExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : FusionableExecution(backend, "pooling", "pooling") {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    mPoolParams    = op->main_as_Pool();
    mPoolType      = mPoolParams->type();

    mStrides[0] = mPoolParams->strideY();
    mStrides[1] = mPoolParams->strideX();
    mKernels[0] = mPoolParams->kernelY();
    mKernels[1] = mPoolParams->kernelX();

    mPaddings[0] = mPoolParams->padY() * 2;
    mPaddings[1] = mPoolParams->padX() * 2;
    mPadType     = mPoolParams->padType();
    if (mPadType == PoolPadType_VALID) {
        mPaddings[0] = 0;
        mPaddings[1] = 0;
    }
    std::set<std::string> buildOptions;
    auto runtime           = mOpenCLBackend->getOpenCLRuntime();

    if (mPoolType == PoolType_AVEPOOL) {
        mBuildOptions.emplace("-DPOOL_AVG");
    }
    mKernel           = runtime->buildKernel(mProgramName, mKernelName, buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
}
bool PoolExecution::fusionable() const {
    return false;
}

ErrorCode PoolExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    uint32_t argIdx = 0;
    return onPrepare(inputs, outputs, nullptr, argIdx, {});
}

ErrorCode PoolExecution::onPrepare(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, cl::Kernel* kernel, uint32_t& argIdx, std::vector<uint32_t> offset) {
    #ifdef LOG_VERBOSE
    MNN_PRINT("start PoolExecution onResize !\n");
#endif

    if (kernel == nullptr) {
        kernel = &mKernel;
    } else {
        uint32_t offsetArgs[2] = {offset[0], offset[1]};
        kernel->setArg(argIdx++, sizeof(offsetArgs), offsetArgs);
    }

    auto input  = inputs[0];
    auto output = outputs[0];

    if (mPoolParams->isGlobal()) {
        std::vector<int> inputShape = tensorShapeFormat(inputs[0]);
        mKernels                    = {inputShape.at(1), inputShape.at(2)};
        mStrides                    = {inputShape.at(1), inputShape.at(2)};
        mPaddings                   = {0, 0};
    }

    if (mPadType == PoolPadType_SAME) {
        int padNeededHeight = std::max(0, (output->height() - 1) * mStrides[0] + mKernels[0] - input->height());
        int padNeededWidth  = std::max(0, (output->width() - 1) * mStrides[1] + mKernels[1] - input->width());

        mPaddings[0] = padNeededHeight;
        mPaddings[1] = padNeededWidth;
    }

    MNN_ASSERT(mDilations[0] == 1 && mDilations[1] == 1);

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    const int batch        = outputShape.at(0);
    const int outputHeight = outputShape.at(1);
    const int outputWidth  = outputShape.at(2);
    const int channels     = outputShape.at(3);

    const int inputHeight = inputShape.at(1);
    const int inputWidth  = inputShape.at(2);

    int channelBlocks = (channels + 3) / 4;

    mGlobalWorkSize = {
        static_cast<uint32_t>(channelBlocks * outputWidth),
        static_cast<uint32_t>(batch * outputHeight)
    };

    int inputImageShape[2] = {inputHeight, inputWidth};
    int paddingShape[2]    = {mPaddings[0] / 2, mPaddings[1] / 2};
    int strideShape[2]     = {mStrides[0], mStrides[1]};
    int kernelShape[2]     = {mKernels[0], mKernels[1]};

    mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime());

    kernel->setArg(argIdx++, mGlobalWorkSize[0]);
    kernel->setArg(argIdx++, mGlobalWorkSize[1]);
    kernel->setArg(argIdx++, openCLImage(input));
    kernel->setArg(argIdx++, sizeof(inputImageShape), inputImageShape);
    kernel->setArg(argIdx++, static_cast<int32_t>(outputHeight));
    kernel->setArg(argIdx++, static_cast<int32_t>(outputWidth));
    kernel->setArg(argIdx++, sizeof(paddingShape), paddingShape);
    kernel->setArg(argIdx++, sizeof(strideShape), strideShape);
    kernel->setArg(argIdx++, sizeof(kernelShape), kernelShape);
    kernel->setArg(argIdx++, openCLImage(output));
#ifdef LOG_VERBOSE
    MNN_PRINT("end PoolExecution onResize !\n");
#endif
    return NO_ERROR;
}

ErrorCode PoolExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start PoolExecution onExecute !\n");
#endif
    
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize,
                       mOpenCLBackend->getOpenCLRuntime(), &event);
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    
#else
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize,
                       mOpenCLBackend->getOpenCLRuntime());
#endif
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end PoolExecution onExecute !\n");
#endif
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<PoolExecution>> __Pool_op(OpType_Pooling);
} // namespace OpenCL
} // namespace MNN
