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
    float* kernelBufferPtr = (float*)mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mKernelBuffer.get()), true, CL_MAP_WRITE,
                                                                                0, filterBuffer->size(), nullptr, nullptr, &error);

    int outputChannel4         = UP_DIV(outputChannel, UNIT);
    int inputChannel4      = UP_DIV(mInputDepth, UNIT);

   if(kernelBufferPtr != nullptr && error == CL_SUCCESS) {
        ::memset(kernelBufferPtr, 0, filterBuffer->size());
        int cur             = 0;
        const float *filterDataPtr = conv2dParams->weight()->data();

        //weight : oc ic -> oc/4 ic/4 ic4 oc4
        //weight image : oc_4, ic_4 * ic4 oc4
        int alignedWeightSize = inputChannel4 * kernelWidth * kernelHeight * UNIT2;
        for (int b = 0; b < outputChannel; ++b) {
            int b_4      = b / UNIT;
            float *dst_b = kernelBufferPtr + b_4 * alignedWeightSize;
            int mx       = b % UNIT;
            for (int d = 0; d < mInputDepth; ++d) {
                int my       = d % UNIT;
                int d_4      = d / UNIT;
                float *dst_d = dst_b + d_4 * kernelWidth * kernelHeight * UNIT2;
                for (int y = 0; y < kernelHeight; ++y) {
                    float *dst_y = dst_d + y * kernelWidth * UNIT2;
                    for (int x = 0; x < kernelWidth; ++x) {
                        float *dst_x          = dst_y + x * UNIT2;
                        if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
                            ((half_float::half*)dst_x)[UNIT * my + mx] = (half_float::half)(filterDataPtr[cur++]);
                        } else {
                            dst_x[UNIT * my + mx] = filterDataPtr[cur++];
                        }
                    }
                }
            }
        }
        
        MNN_PRINT("Opencl Conv weight");
        for(int i = 0 ; i < filterBuffer->size() / sizeof(float); i++) {
            MNN_PRINT("%f ", kernelBufferPtr[i]);
        }
        MNN_PRINT("\n");
    }
    else {
        MNN_ERROR("Map error ptrCL == nullptr \n");
    }

    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*(mKernelBuffer.get()), kernelBufferPtr);

    std::vector<int> filterImageShape{inputChannel4 * UNIT * kernelWidth * kernelHeight, outputChannel4};
    mFilter.reset(Tensor::createDevice<float>({1, filterImageShape[0], 1, filterImageShape[1]}));
    mOpenCLBackend->onAcquireBuffer(mFilter.get(), Backend::STATIC);
    MNN::OpenCL::ImageBufferConvertor imageBufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
    imageBufferConvertor.convertBufferToImage(filterBuffer.get(), MNN::OpenCL::IM2COL_CONV2D_FILTER, mFilter.get());

    //bias
    mBiasBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                        ALIGN_UP4(mConv2dCommonParams->outputCount()) * sizeof(float)));
    float* biasBufferPtr = (float*)mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mBiasBuffer.get()), true, CL_MAP_WRITE,
                                                                                    0, ALIGN_UP4(mConv2dCommonParams->outputCount()) * sizeof(float), nullptr, nullptr, &error);

    if(biasBufferPtr != nullptr){
        ::memset(biasBufferPtr, 0, ALIGN_UP4(mConv2dCommonParams->outputCount()) * sizeof(float));
        ::memcpy(biasBufferPtr, conv2dParams->bias()->data(),
                 conv2dParams->bias()->size() * sizeof(float));
                 
        MNN_PRINT("Opencl Conv bias");
        for(int i = 0 ; i < conv2dParams->bias()->size(); i++) {
            MNN_PRINT("%f ", biasBufferPtr[i]);
        }
        MNN_PRINT("\n");
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

    mPaddings[0] = std::max(mPaddings[0], 0);
    mPaddings[1] = std::max(mPaddings[1], 0);

    std::vector<std::string> im2colPrefix;
    std::vector<std::string> gemmPrefix;
    std::vector<std::string> col2imPrefix;

    if (mConv2dCommonParams->relu()) {
        im2colPrefix.push_back("#define RELU");
        gemmPrefix.push_back("#define RELU");
        col2imPrefix.push_back("#define RELU");
    }
    if (mConv2dCommonParams->relu6()) {
        im2colPrefix.push_back("#define RELU6");
        gemmPrefix.push_back("#define RELU6");
        col2imPrefix.push_back("#define RELU6");
    }

    int ob = outputs[0]->batch();
    int oc = outputs[0]->channel();
    int oh = outputs[0]->height();
    int ow = outputs[0]->width();

    int ic = inputs[0]->channel();

    obxohxow_4  = UP_DIV(ob*oh*ow, 4);

    int fw                = mConv2dCommonParams->kernelX();
    int fh                = mConv2dCommonParams->kernelY();

    auto imageChannelType = mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16() ? CL_HALF_FLOAT : CL_FLOAT;

    //input : temp image : (ib*oh*ow)/ 4, ic/4*(ib*oh*ow)%4*ic4
    //output : temp image : oc/4 * (ob*oh*ow)%4, (ob*oh*ow)/4 * oc4
    mSrcTexture = std::shared_ptr<cl::Image2D>(new cl::Image2D(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, imageChannelType),
                                                   UP_DIV(ic, 4) * UNIT * fw * fh, obxohxow_4));

    mDstTexture = std::shared_ptr<cl::Image2D>(new cl::Image2D(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, imageChannelType),
                                                   obxohxow_4, UP_DIV(oc, 4) * UNIT));

    // Clear texture?

    //if (true == mIsConv1x1) {
    //    setLocalSize(im2colPrefix, mIm2colSize, 8, 8, 1);
    //    mIm2ColProgram = mGLBackend->getProgram("image2col1x1", glsl_im2col1x1_glsl, im2colPrefix);
    //}else{
    //    setLocalSize(im2colPrefix, mIm2colSize, 8, 8, 1);
    //    mIm2ColProgram = mGLBackend->getProgram("image2col", glsl_im2col_glsl, im2colPrefix);
    //}
//
    //setLocalSize(gemmPrefix, mGemmSize, 8, 8, 1);
    //mGemm16x16Program = mGLBackend->getProgram("gemm16x16", glsl_gemm16x16_glsl, gemmPrefix);
    //setLocalSize(col2imPrefix, mCol2imSize, 8, 8, 1);
    //mCol2ImProgram = mGLBackend->getProgram("col2image", glsl_col2im_glsl, col2imPrefix);
    //if (!mIsConv1x1) {
    //    mImage2ColUniform = [=]() {
    //        glUniform2i(2, mPadX, mPadY);
    //        glUniform2i(3, mCommon->kernelX(), mCommon->kernelY());
    //        glUniform2i(4, mCommon->strideX(), mCommon->strideY());
    //        glUniform2i(5, mCommon->dilateX(), mCommon->dilateY());
    //    };
    //}
//
    return NO_ERROR;
}

ErrorCode ConvIm2ColExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    auto input         = inputs[0];
    auto output        = outputs[0];
    auto inputTexture  = input->deviceId();
    auto outputTexture = output->deviceId();

    int iw = input->width();
    int ih = input->height();
    int ic = input->channel();
    int ib = input->batch();

    int ow = output->width();
    int oh = output->height();
    int oc = output->channel();
    int ob = output->batch();

    int ic_4 = UP_DIV(ic, 4);
    int oc_4 = UP_DIV(oc, 4);

    //        image2col
    //{
    //    mIm2ColProgram->useProgram();
    //    glBindImageTexture(0, mSrcTexture->id(), 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());
    //    {
    //        int texId = 0;
    //        glActiveTexture(GL_TEXTURE0 + texId);
    //        glUniform1i(1, texId);
    //        glBindTexture(GL_TEXTURE_3D, inputTexture);
    //        OPENGL_CHECK_ERROR;
    //    }
//
    //    if (mIsConv1x1) {
    //        glUniform1i(5, ic_4);
    //        glUniform1i(6, ow);
    //        glUniform1i(7, oh);
    //    }else{
    //        mImage2ColUniform();
    //        glUniform4i(6, iw, ih, ic_4, 1);
    //        glUniform4i(7, ow, oh, oc_4, 1);
    //    }
    //    OPENGL_CHECK_ERROR;
    //    ((GLBackend *)backend())->compute(UP_DIV(ow, mIm2colSize[0]), UP_DIV(oh, mIm2colSize[1]), UP_DIV(ic_4*ib, mIm2colSize[2]));
    //    OPENGL_CHECK_ERROR;
    //}
//
    ////gemm
    //{
    //    mGemm16x16Program->useProgram();
    //    glBindImageTexture(0, mDstTexture->id(), 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());
    //    OPENGL_CHECK_ERROR;
    //    glBindImageTexture(1, mSrcTexture->id(), 0, GL_TRUE, 0, GL_READ_ONLY, ((GLBackend *)backend())->getTextrueFormat());
    //    glBindImageTexture(2, mKernelTexture->id(), 0, GL_TRUE, 0, GL_READ_ONLY, ((GLBackend *)backend())->getTextrueFormat());
    //    glUniform2i(3, obxohxow_4, oc_4);
    //    if (mIsConv1x1) {
    //        glUniform1i(4, ic_4);
    //    }else{
    //        glUniform1i(4, ic_4*mCommon->kernelX()*mCommon->kernelY());
    //    }
    //    OPENGL_CHECK_ERROR;
    //    ((GLBackend *)backend())->compute(UP_DIV(obxohxow_4, mGemmSize[0]), UP_DIV(oc_4, mGemmSize[1]), 1);
    //    OPENGL_CHECK_ERROR;
    //}
//
    ////col2image
    //{
    //    mCol2ImProgram->useProgram();
    //    OPENGL_CHECK_ERROR;
    //    glBindImageTexture(0, outputTexture, 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());
    //    {
    //        int texId = 0;
    //        glActiveTexture(GL_TEXTURE0 + texId);
    //        glUniform1i(1, texId);
    //        glBindTexture(GL_TEXTURE_2D, mDstTexture->id());
    //        OPENGL_CHECK_ERROR;
    //    }
    //    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mBiasBuffer->getId());
    //    OPENGL_CHECK_ERROR;
    //    glUniform3i(3, ow, oh, oc_4);
    //    OPENGL_CHECK_ERROR;
    //    ((GLBackend *)backend())->compute(UP_DIV(ow, mCol2imSize[0]), UP_DIV(oh, mCol2imSize[1]), UP_DIV(oc_4*ob, mCol2imSize[2]));
    //    OPENGL_CHECK_ERROR;
    //}
//
    return NO_ERROR;
}

}
}