//
//  DepthwiseConvExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/DepthwiseConvExecution.hpp"
#include "core/Macro.h"
#include <string.h>
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

DepthwiseConvExecution::DepthwiseConvExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : ConvCommonExecution(op->main_as_Convolution2D(), op, backend, "depthwise_conv2d", "depthwise_conv2d") {
    mOpenCLBackend      = static_cast<OpenCLBackend *>(backend);
    mCon2dParams        = op->main_as_Convolution2D();
    mConv2dCommonParams = mCon2dParams->common();
    mStrides            = {mConv2dCommonParams->strideY(), mConv2dCommonParams->strideX()};
    mDilations          = {mConv2dCommonParams->dilateY(), mConv2dCommonParams->dilateX()};
    mName = "DepthwiseConv2D";

    mPaddings[0]    = mConv2dCommonParams->padY() * 2;
    mPaddings[1]    = mConv2dCommonParams->padX() * 2;
    PadMode padMode = mConv2dCommonParams->padMode();
    if (padMode == PadMode_VALID) {
        mPaddings[0] = 0;
        mPaddings[1] = 0;
    }

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
    if(ptrCL != nullptr && error == CL_SUCCESS){
        ::memcpy(ptrCL, filterDataPtr, filterBuffer->size());
    }else{
        MNN_ERROR("Map error ptrCL == nullptr \n");
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);

    mOpenCLBackend->onAcquireBuffer(mFilter.get(), Backend::STATIC);

    MNN::OpenCL::ImageBufferConvertor imageBufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
    imageBufferConvertor.convertBufferToImage(filterBuffer.get(), MNN::OpenCL::DW_CONV2D_FILTER, mFilter.get());
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    std::set<std::string> buildOptions;
    
    if (mConv2dCommonParams->strideX() == 1 && mConv2dCommonParams->strideY() == 1 &&
        mConv2dCommonParams->dilateX() == 1 && mConv2dCommonParams->dilateY() == 1) {
        mKernelName = "depthwise_conv2d_s1";
        mName = "DepthwiseConv2D_s1";
    }

    if (mConv2dCommonParams->relu() == true) {
        buildOptions.emplace("-DRELU");
    } else if (mConv2dCommonParams->relu6() == true) {
        buildOptions.emplace("-DRELU6");
    }

    mKernel           = runtime->buildKernel(mProgramName, mKernelName, buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
}

DepthwiseConvExecution::~DepthwiseConvExecution() {
    mOpenCLBackend->onReleaseBuffer(mFilter.get(), Backend::STATIC);
}

ErrorCode DepthwiseConvExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    uint32_t argIdx = 0;
    return onPrepare(inputs, outputs, nullptr, argIdx, {});
}

ErrorCode DepthwiseConvExecution::onPrepare(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, 
                            cl::Kernel* kernel, uint32_t& argIdx, std::vector<uint32_t> offset) {

    if (kernel == nullptr) {
        kernel = &mKernel;
    } else {
        uint32_t offsetArgs[2] = {offset[0], offset[1]};
        kernel->setArg(argIdx++, sizeof(offsetArgs), offsetArgs);
    }

    auto input                   = inputs[0];
    auto output                  = outputs[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), 4) * UP_DIV(outputShape.at(2), 4)),
                       static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))};

    if (mConv2dCommonParams->padMode() == PadMode_SAME) {
        int kernelHeightSize = (mConv2dCommonParams->kernelY() - 1) * mConv2dCommonParams->dilateY() + 1;
        int padNeededHeight =
            (output->height() - 1) * mConv2dCommonParams->strideY() + kernelHeightSize - input->height();
        int kernelWidthSize = (mConv2dCommonParams->kernelX() - 1) * mConv2dCommonParams->dilateX() + 1;
        int padNeededWidth =
            (output->width() - 1) * mConv2dCommonParams->strideX() + kernelWidthSize - input->width();

        mPaddings[0] = padNeededHeight;
        mPaddings[1] = padNeededWidth;
    }

    const int outputHeight = outputShape.at(1);
    const int outputWidth  = outputShape.at(2);

    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);

    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    const int filterHeight       = mCon2dParams->common()->kernelY();
    const int filterWidth        = mCon2dParams->common()->kernelX();

    int inputImageShape[2]  = {inputHeight, inputWidth};
    int outputImageShape[2] = {outputHeight, outputWidth};
    int strideShape[2]      = {mStrides[0], mStrides[1]};
    int paddingShape[2]     = {mPaddings[0] / 2, mPaddings[1] / 2};
    int kernelShape[2]      = {filterHeight, filterWidth};
    int dilationShape[2]    = {mDilations[0], mDilations[1]};

    kernel->setArg(argIdx++, mGlobalWorkSize[0]);
    kernel->setArg(argIdx++, mGlobalWorkSize[1]);
    kernel->setArg(argIdx++, openCLImage(input));
    kernel->setArg(argIdx++, openCLImage(mFilter.get()));
    kernel->setArg(argIdx++, openCLImage(mBias.get()));
    kernel->setArg(argIdx++, openCLImage(output));
    kernel->setArg(argIdx++, sizeof(inputImageShape), inputImageShape);
    kernel->setArg(argIdx++, static_cast<int>(inputChannelBlocks));
    kernel->setArg(argIdx++, sizeof(outputImageShape), outputImageShape);
    kernel->setArg(argIdx++, sizeof(kernelShape), kernelShape);
    kernel->setArg(argIdx++, sizeof(paddingShape), paddingShape);
    if (mStrides[0] != 1 || mStrides[1] != 1 || mDilations[0] != 1 || mDilations[1] != 1) {
        kernel->setArg(argIdx++, sizeof(dilationShape), dilationShape);
        kernel->setArg(argIdx++, sizeof(strideShape), strideShape);
    }
    
    mLocalWorkSize  = depthwiseConvLocalWS(mGlobalWorkSize, mMaxWorkGroupSize);

    return NO_ERROR;
}

std::vector<uint32_t> DepthwiseConvExecution::depthwiseConvLocalWS(const std::vector<uint32_t> &gws,
                                                                   const uint32_t maxWorkGroupSize) {
#ifdef MNN_OPENCL_LWS_TUNE
    MNN_ASSERT(gws.size() == 2);

    auto& tunedLws = mOpenCLBackend->getOpenCLRuntime()->tunedLwsMap();
    std::pair<std::string, std::vector<uint32_t>> info = std::make_pair(mName + "depthwiseConvLocalWS", gws);
    if (tunedLws.find(info) != tunedLws.end()) {
        //printf("depthwiseConvLocalWS Found! gws:%d %d lws:%d %d\n", gws[0], gws[1], tunedLws[info][0], tunedLws[info][1]);
        return tunedLws[info];
    }
    
    std::vector<uint32_t> lws(3, 1);
    std::vector<uint32_t> lws_prefer(4, 1);
    int min_cost = INT_MAX;

    while(lws[1] <= gws[1]) {
        lws[0] = 1;
        while(lws[0] <= gws[0]) {
            if(lws[0]*lws[1] <= maxWorkGroupSize) {
                cl::Event event;
                std::vector<uint32_t> internalGlobalWS(2, 1);
                for (size_t i = 0; i < gws.size(); ++i) {
                    internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
                }
                cl_int error = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueNDRangeKernel(
                                mKernel, cl::NullRange,
                                cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]),
                                cl::NDRange(lws[0], lws[1]),
                                nullptr, &event);
                MNN_CHECK_CL_SUCCESS(error);

                int cost_time = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
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

    if (tunedLws.find(info) == tunedLws.end()) {
        //printf("depthwiseConvLocalWS %d Insert! gws:%d %d, lws:%d %d\n", (int)tunedLws.size(), gws[0], gws[1], lws_prefer[0], lws_prefer[1]);
        tunedLws.insert(std::make_pair(info, lws_prefer));
    }
    
    return lws_prefer;
#else
    
    uint32_t deviceComputeUnits = mOpenCLBackend->getOpenCLRuntime()->deviceComputeUnits();
    std::vector<uint32_t> lws(4, 0);

    int coreNum   = deviceComputeUnits * 4;
    int remain    = gws[0] % coreNum;
    int groupSize = gws[0] / coreNum;
    if (remain == 0) {
        lws[0] = groupSize;
    } else {
        while (groupSize) {
            int remain = gws[0] % groupSize;
            if (remain == 0 && groupSize <= maxWorkGroupSize) {
                lws[0] = groupSize;
                break;
            }
            groupSize--;
        }
    }
    lws[0] = std::max<uint32_t>(std::min<uint32_t>(maxWorkGroupSize, lws[0]), 1);

    remain    = gws[1] % coreNum;
    groupSize = gws[1] / coreNum;
    if (remain == 0) {
        lws[1] = groupSize;
    } else {
        while (groupSize) {
            int remain = gws[1] % groupSize;
            if (remain == 0) {
                lws[1] = groupSize;
                break;
            }
            groupSize--;
        }
    }
    lws[1] = std::max<uint32_t>(std::min<uint32_t>(maxWorkGroupSize / lws[0], lws[1]), 1);

    return lws;
#endif
}

ErrorCode DepthwiseConvExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start DepthwiseConvExecution onExecute !\n");
#endif

#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize,
                mOpenCLBackend->getOpenCLRuntime(),
                &event);
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    
#else
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize,
                mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end DepthwiseConvExecution onExecute !\n");
#endif
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<DepthwiseConvExecution>> __DepthwiseConv_op(OpType_ConvolutionDepthwise);

} // namespace OpenCL
} // namespace MNN
