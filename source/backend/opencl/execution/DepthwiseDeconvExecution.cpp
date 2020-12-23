//
//  DepthwiseDeconvExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/DepthwiseDeconvExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

DepthwiseDeconvExecution::DepthwiseDeconvExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op,
                                                   Backend *backend)
    : ConvCommonExecution(op->main_as_Convolution2D(), op, backend, "depthwise_deconv2d", "depthwise_deconv2d") {
    mOpenCLBackend      = static_cast<OpenCLBackend *>(backend);
    mCon2dParams        = op->main_as_Convolution2D();
    mConv2dCommonParams = mCon2dParams->common();
    mStrides            = {mConv2dCommonParams->strideY(), mConv2dCommonParams->strideX()};
    mDilations          = {mConv2dCommonParams->dilateY(), mConv2dCommonParams->dilateX()};

    mPaddings[0]    = mConv2dCommonParams->padY() * 2;
    mPaddings[1]    = mConv2dCommonParams->padX() * 2;
    PadMode padMode = mConv2dCommonParams->padMode();
    if (padMode == PadMode_VALID) {
        mPaddings[0] = 0;
        mPaddings[1] = 0;
    }
    MNN_ASSERT(mStrides[0] > 0 && mStrides[1] > 0);

    int kernelWidth   = mConv2dCommonParams->kernelX();
    int kernelHeight  = mConv2dCommonParams->kernelY();
    int outputChannel = mConv2dCommonParams->outputCount();

    std::vector<int> filterShape{1, outputChannel, kernelHeight, kernelWidth};
    std::vector<int> filterImageShape{(int)kernelHeight * kernelWidth, (int)UP_DIV(outputChannel, 4)};
    const float *filterDataPtr = mCon2dParams->weight()->data();
    mFilter.reset(Tensor::createDevice<float>({1, filterImageShape[1], 1, 4 * filterImageShape[0]}));
    std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>(filterShape));
    cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                              filterBuffer->size());
    filterBuffer->buffer().device = (uint64_t)(&filterBufferCL);
    cl_int error;
    auto ptrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE,
                                                                                     0, filterBuffer->size(), nullptr, nullptr, &error);
    if(nullptr != ptrCL && error == CL_SUCCESS){
        ::memcpy(ptrCL, filterDataPtr, filterBuffer->size());
    }else{
        MNN_ERROR("Map error ptrCL == nullptr \n");
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);
    mOpenCLBackend->onAcquireBuffer(mFilter.get(), Backend::STATIC);

    MNN::OpenCL::ImageBufferConvertor imageBufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
    imageBufferConvertor.convertBufferToImage(filterBuffer.get(), MNN::OpenCL::DW_CONV2D_FILTER, mFilter.get());
    std::set<std::string> buildOptions;
    if (mConv2dCommonParams->relu() == true) {
        buildOptions.emplace("-DRELU");
    } else if (mConv2dCommonParams->relu6() == true) {
        buildOptions.emplace("-DRELU6");
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    mKernel           = runtime->buildKernel(mProgramName, mKernelName, buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
}
DepthwiseDeconvExecution::~DepthwiseDeconvExecution() {
    mOpenCLBackend->onReleaseBuffer(mFilter.get(), Backend::STATIC);
}
ErrorCode DepthwiseDeconvExecution::onResize(const std::vector<Tensor *> &inputs,
                                             const std::vector<Tensor *> &outputs) {
    uint32_t argIdx = 0;
    return onPrepare(inputs, outputs, nullptr, argIdx, {});
}


ErrorCode DepthwiseDeconvExecution::onPrepare(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, 
                                cl::Kernel* kernel, uint32_t& argIdx, std::vector<uint32_t> offset) {
    if (kernel == nullptr) {
        kernel = &mKernel;
    } else {
        uint32_t offsetArgs[2] = {offset[0], offset[1]};
        kernel->setArg(argIdx++, sizeof(offsetArgs), offsetArgs);
    }

    auto input  = inputs[0];
    auto output = outputs[0];

    if (mConv2dCommonParams->padMode() == PadMode_SAME) {
        int padNeededHeight =
            (output->height() - 1) * mConv2dCommonParams->strideY() + mConv2dCommonParams->kernelY() - input->height();
        int padNeededWidth =
            (output->width() - 1) * mConv2dCommonParams->strideX() + mConv2dCommonParams->kernelX() - input->width();

        mPaddings[0] = padNeededHeight;
        mPaddings[1] = padNeededWidth;
    }
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    const int outputBatch    = outputShape.at(0);
    const int outputHeight   = outputShape.at(1);
    const int outputWidth    = outputShape.at(2);
    const int outputChannels = outputShape.at(3);

    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);

    const int strideHeight = mStrides[0];
    const int strideWidth  = mStrides[1];

    const int channelBlocks = UP_DIV(outputChannels, 4);

    const int paddingHeight = UP_DIV(mPaddings[0], 2);
    const int paddingWidth  = UP_DIV(mPaddings[1], 2);

    const int alignHeight = strideHeight - 1 - paddingHeight;
    const int alignWidth  = strideWidth - 1 - paddingWidth;

    const int filterHeight = mConv2dCommonParams->kernelY();
    const int filterWidth  = mConv2dCommonParams->kernelX();
    const int kernelSize   = filterHeight * filterWidth;

    mGlobalWorkSize        = {static_cast<uint32_t>(channelBlocks), static_cast<uint32_t>(outputWidth),
            static_cast<uint32_t>(outputHeight * outputBatch)};

    int inputImageShape[2]  = {inputHeight, inputWidth};
    int outputImageShape[2] = {outputHeight, outputWidth};
    int strideShape[2]      = {strideHeight, strideWidth};
    int paddingShape[2]     = {paddingHeight, paddingWidth};
    int alignShape[2]       = {alignHeight, alignWidth};
    int kernelShape[2]      = {filterHeight, filterWidth};

    kernel->setArg(argIdx++, mGlobalWorkSize[0]);
    kernel->setArg(argIdx++, mGlobalWorkSize[1]);
    kernel->setArg(argIdx++, mGlobalWorkSize[2]);

    kernel->setArg(argIdx++, openCLImage(input));
    kernel->setArg(argIdx++, openCLImage(mFilter.get()));
    kernel->setArg(argIdx++, openCLImage(mBias.get()));
    kernel->setArg(argIdx++, openCLImage(output));
    kernel->setArg(argIdx++, sizeof(inputImageShape), inputImageShape);
    kernel->setArg(argIdx++, sizeof(outputImageShape), outputImageShape);
    kernel->setArg(argIdx++, sizeof(strideShape), strideShape);
    kernel->setArg(argIdx++, sizeof(alignShape), alignShape);
    kernel->setArg(argIdx++, sizeof(paddingShape), paddingShape);
    kernel->setArg(argIdx++, sizeof(kernelShape), kernelShape);
    kernel->setArg(argIdx++, static_cast<int32_t>(kernelSize));
    kernel->setArg(argIdx++, static_cast<int32_t>(channelBlocks));
    mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime());

    return NO_ERROR;
}

ErrorCode DepthwiseDeconvExecution::onExecute(const std::vector<Tensor *> &inputs,
                                              const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start DepthwiseDeconvExecution onExecute !\n");
#endif
    
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                       mOpenCLBackend->getOpenCLRuntime(),
                       &event);
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    
#else
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                       mOpenCLBackend->getOpenCLRuntime());
#endif
    
#ifdef LOG_VERBOSE
    MNN_PRINT("End DepthwiseDeconvExecution onExecute !\n");
#endif
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<DepthwiseDeconvExecution>> __DepthwiseDeconv_op(OpType_DeconvolutionDepthwise);

} // namespace OpenCL
} // namespace MNN
