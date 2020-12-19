//
//  ConvExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvExecution.hpp"
#include "MultiInputConvExecution.hpp"
#include "ConvWinograd.hpp"
#include "core/ConvolutionCommon.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

#include "half.hpp"

#define UNIT 4
namespace MNN {
namespace OpenCL {

std::vector<uint32_t> ConvExecution::conv2d1x1LocalWSOpt(std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize) {
    
#ifdef MNN_OPENCL_LWS_TUNE
    MNN_ASSERT(gws.size() == 2);
    
    auto& tunedLws = mOpenCLBackend->getOpenCLRuntime()->tunedLwsMap();
    std::pair<std::string, std::vector<uint32_t>> info = std::make_pair(mName + "conv2d1x1LocalWSOpt", gws);
    if (tunedLws.find(info) != tunedLws.end()) {
        //printf("conv2d1x1LocalWSOpt Found! gws:%d %d lws:%d %d\n", gws[0], gws[1], tunedLws[info][0], tunedLws[info][1]);
        return tunedLws[info];
    }
        
    std::vector<uint32_t> lws(3, 1);
    std::vector<uint32_t> lws_prefer(4, 1);
    int min_cost = INT_MAX;

    while(lws[1] <= gws[1]*2 || lws[1] <= 4) {
        lws[0] = 1;
        while(lws[0] <= gws[0]*2  || lws[0] <= 4) {
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
        //printf("conv2d1x1LocalWSOpt %d Insert! gws:%d %d, lws:%d %d\n", (int)tunedLws.size(), gws[0], gws[1], lws_prefer[0], lws_prefer[1]);
        tunedLws.insert(std::make_pair(info, lws_prefer));
    }

    return lws_prefer;
#else
    
    uint32_t deviceComputeUnits = mOpenCLBackend->getOpenCLRuntime()->deviceComputeUnits();

    std::vector<uint32_t> lws(4, 1);

    int coreNum   = deviceComputeUnits * 2;
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

std::vector<uint32_t> ConvExecution::conv2d1x1LocalWS(std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize) {
    
#ifdef MNN_OPENCL_LWS_TUNE
    MNN_ASSERT(gws.size() == 2);
    
    auto& tunedLws = mOpenCLBackend->getOpenCLRuntime()->tunedLwsMap();
    std::pair<std::string, std::vector<uint32_t>> info = std::make_pair(mName + "conv2d1x1LocalWS", gws);
    if (tunedLws.find(info) != tunedLws.end()) {
        //printf("conv2d1x1LocalWS Found! gws:%d %d lws:%d %d\n", gws[0], gws[1], tunedLws[info][0], tunedLws[info][1]);
        return tunedLws[info];
    }
    
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
        //printf("conv2d1x1LocalWS %d Insert! gws:%d %d, lws:%d %d\n", (int)tunedLws.size(), gws[0], gws[1], lws_prefer[0], lws_prefer[1]);
        tunedLws.insert(std::make_pair(info, lws_prefer));
    }
    
    return lws_prefer;
    
#else
    uint32_t cu = mOpenCLBackend->getOpenCLRuntime()->deviceComputeUnits();
    int waveSize = 16; //could be 8, 16, 32, 64, 128 in Adreno GPU
    std::vector<uint32_t> lws(4, 0);

    int coreNum   = cu*2;
    int groupSize = ROUND_UP(gws[0] / coreNum, waveSize);

    lws[0] = groupSize;
    lws[0] = std::max<uint32_t>(std::min<uint32_t>(maxWorkGroupSize, lws[0]), 1);

    int remain = ((maxWorkGroupSize - lws[0]) / waveSize) * waveSize;
    groupSize = ROUND_UP(gws[1] / coreNum, waveSize);
    lws[1] = groupSize;
    lws[1] = std::max<uint32_t>(std::min<uint32_t>(remain / lws[0], lws[1]), 1);
    return lws;
#endif
}

std::vector<uint32_t> ConvExecution::conv2dGeneralLocalWS(const std::vector<uint32_t> &gws, const uint32_t kernelSize,
                                                          const uint32_t maxWorkGroupSize) {
#ifdef MNN_OPENCL_LWS_TUNE
    MNN_ASSERT(gws.size() == 2);
    
    auto& tunedLws = mOpenCLBackend->getOpenCLRuntime()->tunedLwsMap();
    std::pair<std::string, std::vector<uint32_t>> info = std::make_pair(mName + "conv2dGeneralLocalWS", gws);
    if (tunedLws.find(info) != tunedLws.end()) {
        //printf("conv2dGeneralLocalWS Found! gws:%d %d lws:%d %d\n", gws[0], gws[1], tunedLws[info][0], tunedLws[info][1]);
        return tunedLws[info];
    }
    
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
        //printf("conv2dGeneralLocalWS %d Insert! gws:%d %d, lws:%d %d\n", (int)tunedLws.size(), gws[0], gws[1], lws_prefer[0], lws_prefer[1]);
        tunedLws.insert(std::make_pair(info, lws_prefer));
    }
    
    return lws_prefer;
#else
    
    uint32_t deviceComputeUnits = mOpenCLBackend->getOpenCLRuntime()->deviceComputeUnits();

    GpuType gpuType = mOpenCLBackend->getOpenCLRuntime()->getGpuType();

    std::vector<uint32_t> lws(4, 0);

    if (gpuType == GpuType::ADRENO) {
        int coreNum   = deviceComputeUnits;
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

        remain    = gws[2] % coreNum;
        groupSize = gws[2] / coreNum;
        if (remain == 0) {
            lws[2] = groupSize;
        } else {
            while (groupSize) {
                int remain = gws[2] % groupSize;
                if (remain == 0) {
                    lws[2] = groupSize;
                    break;
                }
                groupSize--;
            }
        }

        lws[2] = std::max<uint32_t>(std::min<uint32_t>(maxWorkGroupSize / (lws[0] * lws[1]), lws[2]), 1);
    } else {
        lws[0] = deviceComputeUnits * 2;
        lws[1] = 4;
        lws[2] = 1;
    }

    return lws;
#endif
}

ConvCommonExecution::ConvCommonExecution(const Convolution2D *conv2dParams, const MNN::Op *op, Backend *backend, std::string programName, std::string kernelName) 
    : FusionableExecution(backend, programName, kernelName) {
    mName = op->name()->c_str();
    auto openclBackend       = (OpenCLBackend *)backend;
    int biasSize             = conv2dParams->bias()->size();
    const float *biasDataPtr = conv2dParams->bias()->data();
    cl::Buffer biasBuffer(openclBackend->getOpenCLRuntime()->context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                          UP_DIV(biasSize, 4) * 4 * sizeof(float));
    cl_int error;
    auto biasPtrCL = openclBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
        biasBuffer, true, CL_MAP_WRITE, 0, ALIGN_UP4(biasSize) * sizeof(float), nullptr, nullptr, &error);
    if(biasPtrCL != nullptr && error == CL_SUCCESS){
        ::memset(biasPtrCL, 0, ALIGN_UP4(biasSize) * sizeof(float));
        ::memcpy(biasPtrCL, biasDataPtr, biasSize * sizeof(float));
    }else{
        MNN_ERROR("Map error biasPtrCL == nullptr \n");
    }
    openclBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(biasBuffer, biasPtrCL);
    mBias.reset(Tensor::createDevice<float>({1, 1, 1, biasSize}));
    backend->onAcquireBuffer(mBias.get(), Backend::STATIC);
    copyBufferToImage(openclBackend->getOpenCLRuntime(), biasBuffer, openCLImage(mBias.get()), UP_DIV(biasSize, 4), 1);
}
ConvCommonExecution::~ConvCommonExecution() {
    MNN_ASSERT(nullptr != mBias);
    backend()->onReleaseBuffer(mBias.get(), Backend::STATIC);
}

ConvExecution::ConvExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : ConvCommonExecution(op->main_as_Convolution2D(), op, backend, "conv_2d") {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution init !\n");
#endif
    mOpenCLBackend                 = static_cast<OpenCLBackend *>(backend);
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mConv2dCommonParams            = conv2dCommonParams;
    mStrides                       = {conv2dCommonParams->strideY(), conv2dCommonParams->strideX()};
    mDilations                     = {conv2dCommonParams->dilateY(), conv2dCommonParams->dilateX()};
    mName = "Conv2D";


    mPaddings[0]    = conv2dCommonParams->padY() * 2;
    mPaddings[1]    = conv2dCommonParams->padX() * 2;
    if (conv2dCommonParams->pads() != nullptr) {
        MNN_ASSERT(conv2dCommonParams->pads()->size() >= 4);
        mPaddings[0] = conv2dCommonParams->pads()->data()[1] * 2;
        mPaddings[1] = conv2dCommonParams->pads()->data()[0] * 2;
    }
    PadMode padMode = conv2dCommonParams->padMode();
    if (padMode == PadMode_VALID) {
        mPaddings[0] = 0;
        mPaddings[1] = 0;
    }

    int kernelWidth   = conv2dCommonParams->kernelX();
    int kernelHeight  = conv2dCommonParams->kernelY();
    int outputChannel = conv2dCommonParams->outputCount();

    int weightSize             = 0;
    const float *filterDataPtr = nullptr;

    std::shared_ptr<MNN::ConvolutionCommon::Int8Common> quanCommon;
    if (nullptr != conv2dParams->quanParameter()) {
        quanCommon = ConvolutionCommon::load(conv2dParams->quanParameter(), true);
        if (nullptr == quanCommon) {
            MNN_ERROR("Memory not Enough, can't extract IDST Convolution: %s \n", op->name()->c_str());
        }
        if (quanCommon->weightFloat.get() == nullptr) {
            MNN_PRINT("quanCommon->weightFloat.get() == nullptr \n");
        }
        // Back to float
        filterDataPtr = quanCommon->weightFloat.get();
        weightSize    = quanCommon->weightFloat.size();
    }

    if (nullptr == filterDataPtr) {
        weightSize    = conv2dParams->weight()->size();
        filterDataPtr = conv2dParams->weight()->data();
    }
    int inputChannel = weightSize / (kernelWidth * kernelHeight * outputChannel);

    auto gpuType = mOpenCLBackend->getOpenCLRuntime()->getGpuType();

    //select opt conv method
    mKernelName = "conv_2d";
    if (kernelHeight == kernelWidth && kernelHeight == 1 && mConv2dCommonParams->padX() == 0 &&
        mConv2dCommonParams->padY() == 0) {
        mConv1x1Opt = (mStrides[0] == 1 && mStrides[1] == 1 && !(gpuType == GpuType::ADRENO));
#if 0
        if(gpuType == GpuType::ADRENO){
            uint64_t useLocalSize = UNIT*UNIT*4*sizeof(float)*4;
            if(useLocalSize >= mOpenCLBackend->getOpenCLRuntime()->getMaxLocalMem()){
                mUseLocalMem = false;
            }else{
                mKernelName = "conv_2d_1x1_local";
                mUseLocalMem=true;
            }
        }
#endif
        
        if(!mUseLocalMem){
            if(mConv1x1Opt){
                mKernelName = "conv_2d_1x1_mali";
            }else{
                mKernelName = "conv_2d_1x1";
            }
        }
    }

    if(mConv1x1Opt && !mUseLocalMem){
        cl_int error;
        std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({UP_DIV(outputChannel, 4)*4, UP_DIV(inputChannel, 4)*4, kernelWidth, kernelHeight}));
        mKernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, filterBuffer->size()));
        auto kernelBufferPtr = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mKernelBuffer.get()), true, CL_MAP_WRITE,
                                                                                    0, filterBuffer->size(), nullptr, nullptr, &error);
        if(kernelBufferPtr != nullptr && error == CL_SUCCESS){
            ::memset(kernelBufferPtr, 0, filterBuffer->size());
            for(int o = 0; o < outputChannel; o++){
                for(int i = 0 ; i < inputChannel; i++){
                    int bufferIdx = (o/4) * ROUND_UP(inputChannel, 4)*4 + (i/4)*16 + (o%4)*4 + (i%4);
                    int filterIdx = o*inputChannel + i;
                    if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()){
                        ((half_float::half*)kernelBufferPtr)[bufferIdx] = (half_float::half)(filterDataPtr[filterIdx]);
                    }else{
                        ((float*)kernelBufferPtr)[bufferIdx] = (float)(filterDataPtr[filterIdx]);
                    }
                }
            }
        }else{
            MNN_ERROR("Map error ptrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*(mKernelBuffer.get()), kernelBufferPtr);

        //bias
        int biasSize             = conv2dParams->bias()->size();
        const float *biasDataPtr = conv2dParams->bias()->data();
        mBiasBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                            UP_DIV(biasSize, 4) * 4 * sizeof(float)));
        auto biasPtrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
            *(mBiasBuffer.get()), true, CL_MAP_WRITE, 0, ALIGN_UP4(biasSize) * sizeof(float), nullptr, nullptr, &error);
        if(biasPtrCL != nullptr && error == CL_SUCCESS){
            if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()){
                ::memset(biasPtrCL, 0, ALIGN_UP4(biasSize) * sizeof(half_float::half));
                for (int i = 0; i < biasSize; i++)
                {
                    ((half_float::half*)biasPtrCL)[i] = (half_float::half)(biasDataPtr[i]);
                }
            }else{
                ::memset(biasPtrCL, 0, ALIGN_UP4(biasSize) * sizeof(float));
                ::memcpy(biasPtrCL, biasDataPtr, biasSize * sizeof(float));
            }
        }else{
            MNN_ERROR("Map error biasPtrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*(mBiasBuffer.get()), biasPtrCL);
    }else{
        std::vector<int> filterImageShape{(int)inputChannel, (int)(UP_DIV(outputChannel, 4) * kernelWidth * kernelHeight)};
        std::shared_ptr<Tensor> filterBuffer(
            Tensor::createDevice<float>({outputChannel, inputChannel, kernelWidth, kernelHeight}));
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

        mFilter.reset(Tensor::createDevice<float>({1, filterImageShape[1], 1, 4 * filterImageShape[0]}));
        mOpenCLBackend->onAcquireBuffer(mFilter.get(), Backend::STATIC);
        MNN::OpenCL::ImageBufferConvertor imageBufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
        imageBufferConvertor.convertBufferToImage(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mFilter.get());
    }

    // Create Kernel
    mBuildOptions.emplace("-DBIAS");
    if (mConv2dCommonParams->relu()) {
        mBuildOptions.emplace("-DRELU");
    } else if (mConv2dCommonParams->relu6()) {
        mBuildOptions.emplace("-DRELU6");
    }

    mKernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel(mProgramName, mKernelName, mBuildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mKernel));

#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution init !\n");
#endif
}

ConvExecution::~ConvExecution() {
    if(mUseLocalMem || !mConv1x1Opt){
        mOpenCLBackend->onReleaseBuffer(mFilter.get(), Backend::STATIC);
    }
}

ErrorCode ConvExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    uint32_t argIdx = 0;
    return onPrepare(inputs, outputs, nullptr, argIdx, {});
}

ErrorCode ConvExecution::onPrepare(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, cl::Kernel* kernel, uint32_t& argIdx, std::vector<uint32_t> offset) {
    #ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution onResize !\n");
#endif

    if (kernel == nullptr) {
        kernel = &mKernel;
    } else {
        uint32_t offsetArgs[2] = {offset[0], offset[1]};
        kernel->setArg(argIdx++, sizeof(offsetArgs), offsetArgs);
    }

    auto input  = inputs[0];
    auto output = outputs[0];

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    const int height             = outputShape.at(1);
    const int width              = outputShape.at(2);

    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);

    const int inputChannelBlocks = UP_DIV(inputChannels, 4);

    if (mConv2dCommonParams->padMode() == PadMode_SAME) {
        int kernelHeightSize = (mConv2dCommonParams->kernelY() - 1) * mConv2dCommonParams->dilateY() + 1;
        int padNeededHeight = (outputShape.at(1) - 1) * mConv2dCommonParams->strideY() +
                kernelHeightSize - inputShape.at(1);
        int kernelWidthSize = (mConv2dCommonParams->kernelX() - 1) * mConv2dCommonParams->dilateX() + 1;
        int padNeededWidth = (outputShape.at(2) - 1) * mConv2dCommonParams->strideX() + kernelWidthSize -
                             inputShape.at(2);
        mPaddings[0] = padNeededHeight;
        mPaddings[1] = padNeededWidth;

    }

    int kernelHeight = mConv2dCommonParams->kernelY();
    int kernelWidth  = mConv2dCommonParams->kernelX();

    mPaddings[0] = std::max(mPaddings[0], 0);
    mPaddings[1] = std::max(mPaddings[1], 0);

    if (kernelHeight == kernelWidth && kernelHeight == 1 && mPaddings[0] == 0 && mPaddings[1] == 0) {
        if(mConv1x1Opt){
            if(mUseLocalMem){
                // conv_2d_1x1_local
                mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), 4)), static_cast<uint32_t>(UP_DIV(outputShape.at(2), 4)),
                static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))};
                std::vector<uint32_t> lws{UNIT, UNIT, 1};
                mLocalWorkSize = lws;
                kernel->setArg(argIdx++, mGlobalWorkSize[0]);
                kernel->setArg(argIdx++, mGlobalWorkSize[1]);
                kernel->setArg(argIdx++, mGlobalWorkSize[2]);
                kernel->setArg(argIdx++, openCLImage(input));
                kernel->setArg(argIdx++, openCLImage(mFilter.get()));
                kernel->setArg(argIdx++, openCLImage(mBias.get()));
                kernel->setArg(argIdx++, openCLImage(output));
                kernel->setArg(argIdx++, static_cast<int>(inputChannelBlocks));
                kernel->setArg(argIdx++, height);
                kernel->setArg(argIdx++, width);
            }else{
                // conv_2d_1x1_mali
                mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), 4) * UP_DIV(outputShape.at(2), 4)),
                           static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))};
                kernel->setArg(argIdx++, mGlobalWorkSize[0]);
                kernel->setArg(argIdx++, mGlobalWorkSize[1]);
                kernel->setArg(argIdx++, UP_DIV(width, 4));
                kernel->setArg(argIdx++, openCLImage(input));
                kernel->setArg(argIdx++, *mKernelBuffer.get());
                kernel->setArg(argIdx++, *mBiasBuffer.get());
                kernel->setArg(argIdx++, openCLImage(output));
                kernel->setArg(argIdx++, static_cast<int>(inputChannelBlocks));
                kernel->setArg(argIdx++, height);
                kernel->setArg(argIdx++, width);
                
                mLocalWorkSize          = conv2d1x1LocalWSOpt(mGlobalWorkSize, mMaxWorkGroupSize);
            }


        }else{
            // conv_2d_1x1
            mGlobalWorkSize = {
            static_cast<uint32_t>(UP_DIV(outputShape.at(3), 4) * static_cast<uint32_t>(UP_DIV(outputShape.at(2), 4))),
            static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))};
            
            int inputImageShape[2]  = {inputHeight, inputWidth};
            int outputImageShape[2] = {height, width};
            int stideShape[2]       = {mStrides[0], mStrides[1]};
            kernel->setArg(argIdx++, mGlobalWorkSize[0]);
            kernel->setArg(argIdx++, mGlobalWorkSize[1]);
            kernel->setArg(argIdx++, openCLImage(input));
            kernel->setArg(argIdx++, openCLImage(mFilter.get()));
            kernel->setArg(argIdx++, openCLImage(mBias.get()));
            kernel->setArg(argIdx++, openCLImage(output));
            kernel->setArg(argIdx++, sizeof(inputImageShape), inputImageShape);
            kernel->setArg(argIdx++, static_cast<int>(inputChannelBlocks));
            kernel->setArg(argIdx++, sizeof(outputImageShape), outputImageShape);
            kernel->setArg(argIdx++, sizeof(stideShape), stideShape);
            kernel->setArg(argIdx++, UP_DIV(width, 4));
            mLocalWorkSize          = conv2d1x1LocalWS(mGlobalWorkSize, mMaxWorkGroupSize);

        }
    } else {
        mGlobalWorkSize         = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), 4) * UP_DIV(outputShape.at(2), 4)),
                           static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))};

        int inputImageShape[2]  = {inputHeight, inputWidth};
        int outputImageShape[2] = {height, width};
        int kernelShape[2]      = {kernelHeight, kernelWidth};
        int strideShape[2]      = {mStrides[0], mStrides[1]};
        int paddingShape[2]     = {mPaddings[0] / 2, mPaddings[1] / 2};
        int dilationShape[2]    = {mDilations[0], mDilations[1]};
        kernel->setArg(argIdx++, mGlobalWorkSize[0]);
        kernel->setArg(argIdx++, mGlobalWorkSize[1]);
        kernel->setArg(argIdx++, openCLImage(input));
        kernel->setArg(argIdx++, openCLImage(mFilter.get()));
        kernel->setArg(argIdx++, openCLImage(mBias.get()));
        kernel->setArg(argIdx++, openCLImage(output));
        kernel->setArg(argIdx++, sizeof(inputImageShape), inputImageShape);
        kernel->setArg(argIdx++, inputChannelBlocks);
        kernel->setArg(argIdx++, sizeof(outputImageShape), outputImageShape);
        kernel->setArg(argIdx++, sizeof(kernelShape), kernelShape);
        kernel->setArg(argIdx++, sizeof(strideShape), strideShape);
        kernel->setArg(argIdx++, sizeof(paddingShape), paddingShape);
        kernel->setArg(argIdx++, sizeof(dilationShape), dilationShape);
        kernel->setArg(argIdx++, UP_DIV(width, 4));
        
        mLocalWorkSize          = conv2dGeneralLocalWS(mGlobalWorkSize, kernelHeight * kernelWidth, mMaxWorkGroupSize);
        
    }

#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution onResize !\n");
#endif
    return NO_ERROR;
}

ErrorCode ConvExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution onExecute !\n");
#endif
    if(mUseLocalMem){
    #ifdef ENABLE_OPENCL_TIME_PROFILER
        cl::Event event;
        run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                           mOpenCLBackend->getOpenCLRuntime(), &event);
        
        float costTime = mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
        MNN_PRINT("kernel cost:%f    us Conv UseLocalMem\n",costTime);
    #else
        run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                           mOpenCLBackend->getOpenCLRuntime());
    #endif
    }
    
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize,
                mOpenCLBackend->getOpenCLRuntime(), &event);
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us Conv2D\n",costTime);
#else
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize,
                mOpenCLBackend->getOpenCLRuntime());
#endif
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution onExecute !\n");
#endif
    return NO_ERROR;
}

class ConvolutionCreator : public OpenCLBackend::Creator {
public:
    virtual ~ConvolutionCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if (inputs.size() == 3) {
            return new MultiInputConvExecution(op, backend);
        }

        auto conv2D = op->main_as_Convolution2D();
        if (ConvWinograd::valid(conv2D->common(), inputs[0])) {
            return new ConvWinograd(conv2D, op, backend);
        }

        return new ConvExecution(inputs, op, backend);
    }
};

OpenCLCreatorRegister<ConvolutionCreator> __conv_op(OpType_Convolution);

} // namespace OpenCL
} // namespace MNN
