#include "backend/opencl/execution/ConvIm2col.hpp"

#include "half.hpp"

namespace MNN {
namespace OpenCL {


ConvIm2ColExecution::~ConvIm2ColExecution() {
}

#define UNIT 4
#define UNIT2 16
ConvIm2ColExecution::ConvIm2ColExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) 
    : ConvCommonExecution(op->main_as_Convolution2D(), backend), mOpenCLBackend((OpenCLBackend *)backend) {
    const auto *conv2dParams       = op->main_as_Convolution2D();
    mConv2dCommonParams            = conv2dParams->common();

    int kernelWidth   = mConv2dCommonParams->kernelX();
    int kernelHeight  = mConv2dCommonParams->kernelY();
    int outputChannel = mConv2dCommonParams->outputCount();

    mIsConv1x1 = (kernelWidth == 1 && kernelHeight == 1) ? true : false;

    mInputDepth = conv2dParams->weight()->size() * mConv2dCommonParams->group() /
                  kernelWidth / kernelHeight / outputChannel;

    auto totalWeightSize = ALIGN_UP4(outputChannel) * ALIGN_UP4(mInputDepth) * (kernelWidth * kernelHeight);

    cl_int error;
    std::shared_ptr<Tensor> filterBuffer(
        Tensor::createDevice<float>({ALIGN_UP4(outputChannel), ALIGN_UP4(mInputDepth), kernelWidth, kernelHeight}));
    mKernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, filterBuffer->size()));
    void* kernelBufferPtr = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mKernelBuffer.get()), true, CL_MAP_WRITE,
                                                                                0, filterBuffer->size(), nullptr, nullptr, &error);
                                                                                
    int outputChannel4         = UP_DIV(outputChannel, UNIT);
    int inputChannel4      = UP_DIV(mInputDepth, UNIT);

    if(kernelBufferPtr != nullptr && error == CL_SUCCESS) {
        ::memset(kernelBufferPtr, 0, filterBuffer->size());
        int count             = 0;
        const float *filterDataPtr = conv2dParams->weight()->data();

        for (int oc = 0; oc < outputChannel; oc++) {
            // IC * kw * kh
            const int oc_offset = ALIGN_UP4(mInputDepth) * kernelWidth * kernelHeight * oc;
            for (int ic4 = 0; ic4 < inputChannel4; ic4++) {
                const int ic4_offset = kernelWidth * kernelHeight * UNIT * ic4;
                for (int y = 0; y < kernelHeight; ++y) {
                    for (int x = 0; x < kernelWidth; ++x) {
                        int remainingChannel = mInputDepth - ic4 * UNIT;
                        for (int c = 0; c < remainingChannel; c++) {
                            int currentChannel = ic4 * UNIT + c;
                            if (mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16())
                                ((half_float::half *)kernelBufferPtr)[
                                    (oc_offset + ic4_offset + y * kernelWidth + x) * 4 + c
                                ] =  filterDataPtr[currentChannel * kernelHeight * kernelWidth + kernelWidth * y + x];
                            else
                                ((float *)kernelBufferPtr)[
                                    (oc_offset + ic4_offset + y * kernelWidth + x) * 4 + c
                                ] =  filterDataPtr[currentChannel * kernelHeight * kernelWidth + kernelWidth * y + x];
                        }
                    }
                }
            }
        }
    }
    else {
        MNN_ERROR("Map error ptrCL == nullptr \n");
    }

    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*(mKernelBuffer.get()), kernelBufferPtr);

    
    auto imageChannelType = CL_HALF_FLOAT;
    if (mOpenCLBackend->getPrecision() == BackendConfig::Precision_High) {
        imageChannelType = CL_FLOAT;
    }

    mKernel.reset(new cl::Image2D(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, imageChannelType),
                                  ALIGN_UP4(mInputDepth) * kernelWidth * kernelHeight, ALIGN_UP4(outputChannel)));
    copyBufferToImage(mOpenCLBackend->getOpenCLRuntime(), *mKernelBuffer, *mKernel, ALIGN_UP4(mInputDepth) * kernelWidth * kernelHeight, ALIGN_UP4(outputChannel));

    //float* hostPtr = (float*) mOpenCLBackend->readImage(mFilter.get());
    //MNN_PRINT("Opencl Conv weight\n");
    //
    //for (int oc = 0; oc < ALIGN_UP4(outputChannel); oc++) {
    //    const int oc_offset = ALIGN_UP4(mInputDepth) * kernelWidth * kernelHeight * oc;
    //    for (int i = 0; i < ALIGN_UP4(mInputDepth) * kernelWidth * kernelHeight; i++) {
    //        MNN_PRINT("%f ", ((float*)hostPtr)[i + oc_offset]);
    //    }
    //    MNN_PRINT("\n");
    //}

    //bias
    mBiasBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                        ALIGN_UP4(mConv2dCommonParams->outputCount()) * sizeof(float)));
    float* biasBufferPtr = (float*)mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mBiasBuffer.get()), true, CL_MAP_WRITE,
                                                                                    0, ALIGN_UP4(mConv2dCommonParams->outputCount()) * sizeof(float), nullptr, nullptr, &error);

    if(biasBufferPtr != nullptr){
        ::memset(biasBufferPtr, 0, ALIGN_UP4(mConv2dCommonParams->outputCount()) * sizeof(float));
        ::memcpy(biasBufferPtr, conv2dParams->bias()->data(),
                 conv2dParams->bias()->size() * sizeof(float));
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*(mBiasBuffer.get()), biasBufferPtr);
}

ErrorCode ConvIm2ColExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvIm2ColExecution onResize !\n");
#endif

    auto input  = inputs[0];
    auto output = outputs[0];

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    const int height             = outputShape.at(1);
    const int width              = outputShape.at(2);

    const int inputBatch = inputShape.at(0);
    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannel = inputShape.at(3);

    std::vector<int> paddings{0, 0};

    if (mConv2dCommonParams->padMode() == PadMode_SAME) {
        int kernelHeightSize = (mConv2dCommonParams->kernelY() - 1) * mConv2dCommonParams->dilateY() + 1;
        int padNeededHeight = (outputShape.at(1) - 1) * mConv2dCommonParams->strideY() +
                kernelHeightSize - inputShape.at(1);
        int kernelWidthSize = (mConv2dCommonParams->kernelX() - 1) * mConv2dCommonParams->dilateX() + 1;
        int padNeededWidth = (outputShape.at(2) - 1) * mConv2dCommonParams->strideX() + kernelWidthSize -
                             inputShape.at(2);
        paddings[0] = padNeededWidth;
        paddings[1] = padNeededHeight;

    }

    paddings[0] = std::max(paddings[0], 0);
    paddings[1] = std::max(paddings[1], 0);

    std::vector<int> kernels = {mConv2dCommonParams->kernelX(), mConv2dCommonParams->kernelY()};
    std::vector<int> strides = {mConv2dCommonParams->strideX(), mConv2dCommonParams->strideY()};
    std::vector<int> dilations  = {mConv2dCommonParams->dilateX(), mConv2dCommonParams->dilateY()};

    const int outputBatch = outputs[0]->batch();
    const int outputChannel = outputs[0]->channel();
    const int outputHeight = outputs[0]->height();
    const int outputWidth = outputs[0]->width();

    const int inputChannel4 = UP_DIV(inputChannel, 4);
    const int outputChannel4 = UP_DIV(outputChannel, 4);

    obxohxow_4  = UP_DIV(outputBatch * outputHeight * outputWidth, 4);

    int filterWidth                = mConv2dCommonParams->kernelX();
    int filterHeight       = mConv2dCommonParams->kernelY();

    auto imageChannelType = mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16() ? CL_HALF_FLOAT : CL_FLOAT;

    //input : temp image : (ib*oh*ow)/ 4, ic/4*(ib*oh*ow)%4*ic4
    //output : temp image : oc/4 * (ob*oh*ow)%4, (ob*oh*ow)/4 * oc4
    mSrcTexture = std::shared_ptr<cl::Image2D>(new cl::Image2D(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, imageChannelType),
                                                   UP_DIV(inputChannel, 4) * UNIT * filterWidth * filterHeight, obxohxow_4));

    mDstTexture = std::shared_ptr<cl::Image2D>(new cl::Image2D(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, imageChannelType),
                                                   obxohxow_4, UP_DIV(outputChannel, 4) * UNIT));

    // Clear texture?

    std::set<std::string> buildOptions;
    buildOptions.emplace("-DBIAS");
    if (mConv2dCommonParams->relu()) {
        buildOptions.emplace("-DRELU");
    } else if (mConv2dCommonParams->relu6()) {
        buildOptions.emplace("-DRELU6");
    }

    {
        uint32_t idx = 0;
        mIm2colSize = {8, 8, 1};
        mIm2colGlobalSize = {UP_DIV(outputWidth, mIm2colSize[0]), UP_DIV(outputHeight, mIm2colSize[1]), UP_DIV(inputChannel4 * inputBatch, mIm2colSize[2])};
        mIm2ColKernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("image2col", mIsConv1x1 ? "image2col_1x1" : "image2col", buildOptions);
        mIm2ColKernel.setArg(idx++, mIm2colGlobalSize[0]); // gws 0 
        mIm2ColKernel.setArg(idx++, mIm2colGlobalSize[1]); // gws 1
        mIm2ColKernel.setArg(idx++, openCLImage(input));
        mIm2ColKernel.setArg(idx++, mSrcTexture); 
        if (mIsConv1x1) {
            mIm2ColKernel.setArg(idx++, inputChannel4);
            mIm2ColKernel.setArg(idx++, outputWidth);
            mIm2ColKernel.setArg(idx++, outputHeight);
        }else{
            mIm2ColKernel.setArg(idx++, paddings);
            mIm2ColKernel.setArg(idx++, kernels);
            mIm2ColKernel.setArg(idx++, strides);
            mIm2ColKernel.setArg(idx++, dilations);
            mIm2ColKernel.setArg(idx++, std::vector<int>{inputWidth, inputHeight, inputChannel4, 1});
            mIm2ColKernel.setArg(idx++, std::vector<int>{outputWidth, outputHeight, outputChannel4, 1});
        }
    }

    {
        uint32_t idx = 0;
        mGemmSize = {8, 8};
        mGemmGlobalSize = {UP_DIV(obxohxow_4, mGemmSize[0]), UP_DIV(outputChannel4, mGemmSize[1])};
        mGemmKernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm", "gemm_16x16", buildOptions);
        mGemmKernel.setArg(idx++, mGemmGlobalSize[0]);
        mGemmKernel.setArg(idx++, mGemmGlobalSize[1]);
        mGemmKernel.setArg(idx++, mDstTexture);
        mGemmKernel.setArg(idx++, mSrcTexture);
        mGemmKernel.setArg(idx++, mKernel);
        mGemmKernel.setArg(idx++, std::vector<int>{obxohxow_4, outputChannel4});
        if (mIsConv1x1) {
            mGemmKernel.setArg(idx++, inputChannel4);
        }else{
            mGemmKernel.setArg(idx++, inputChannel4 * filterWidth * filterHeight);
        }
    }

    {
        uint32_t idx = 0;
        mCol2imSize = {8, 8, 1};
        mCol2imGlobalSize = {UP_DIV(outputWidth, mCol2imSize[0]), UP_DIV(outputHeight, mCol2imSize[1]), UP_DIV(outputChannel4 * outputBatch, mCol2imSize[2])};
        mCol2ImKernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("col2image", mIsConv1x1 ? "col2image_1x1" : "col2image", buildOptions);
        mCol2ImKernel.setArg(idx++, mCol2imGlobalSize[0]);
        mCol2ImKernel.setArg(idx++, mCol2imGlobalSize[1]);
        mCol2ImKernel.setArg(idx++, mCol2imGlobalSize[2]);
        mCol2ImKernel.setArg(idx++, openCLImage(output));
        mCol2ImKernel.setArg(idx++, mDstTexture);
        mCol2ImKernel.setArg(idx++, mBiasBuffer);
        mCol2ImKernel.setArg(idx++, std::vector<int>{outputWidth, outputHeight, outputChannel4});
    }

    return NO_ERROR;
}

ErrorCode ConvIm2ColExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvIm2ColExecution onExecute !\n");
#endif
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                        mOpenCLBackend->getOpenCLRuntime(), &event);
    
    float costTime = mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%f    us Conv UseLocalMem\n",costTime);
#else
    run3DKernelDefault(mIm2ColKernel, mIm2colGlobalSize, mIm2colSize,
                        mOpenCLBackend->getOpenCLRuntime());
#endif
    
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    runKernel2D(mGemmKernel, mGemmGlobalSize, mGemmSize,
                mOpenCLBackend->getOpenCLRuntime(), &event);
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us Conv2D\n",costTime);
#else
    runKernel2D(mGemmKernel, mGemmGlobalSize, mGemmSize,
                mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mCol2ImKernel, mCol2imGlobalSize, mCol2imSize,
                        mOpenCLBackend->getOpenCLRuntime(), &event);
    
    float costTime = mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%f    us Conv UseLocalMem\n",costTime);
#else
    run3DKernelDefault(mCol2ImKernel, mCol2imGlobalSize, mCol2imSize,
                        mOpenCLBackend->getOpenCLRuntime());
#endif
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvIm2ColExecution onExecute !\n");
#endif
    return NO_ERROR;
}

}
}