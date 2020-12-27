//
//  ConvertExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/ConvertExecution.hpp"
#include "core/Macro.h"
#include "backend/cpu/CPUTensorConvert.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
    namespace OpenCL {

        ConvertExecution::ConvertExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend)
        : FusionableExecution(backend, "convert", "convert") {
            mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
            std::set<std::string> buildOptions;

            mKernel    = mOpenCLBackend->getOpenCLRuntime()->buildKernel(mProgramName, mKernelName, buildOptions);
            mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mKernel));
        }

        ErrorCode ConvertExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
            uint32_t argIdx = 0;
            return onPrepare(inputs, outputs, nullptr, argIdx, {});
        }
        
        ErrorCode ConvertExecution::onPrepare(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, 
                                    cl::Kernel* kernel, uint32_t& argIdx, std::vector<uint32_t> offset) {

            if (kernel == nullptr) {
                kernel = &mKernel;
            } else {
                uint32_t offsetArgs[2] = {offset[0], offset[1]};
                kernel->setArg(argIdx++, sizeof(offsetArgs), offsetArgs);
            }

            Tensor* input  = inputs[0];
            Tensor* output = outputs[0];

            std::vector<int> inputShape  = tensorShapeFormat(input);
            std::vector<int> outputShape = tensorShapeFormat(output);

            const int batch    = inputShape.at(0);
            const int height   = inputShape.at(1);
            const int width    = inputShape.at(2);
            const int channels = inputShape.at(3);

            const int channelBlocks = UP_DIV(channels, 4);

            mGlobalWorkSize = {static_cast<uint32_t>(channelBlocks * width),
                static_cast<uint32_t>(height * batch)};

            kernel->setArg(argIdx++, mGlobalWorkSize[0]);
            kernel->setArg(argIdx++, mGlobalWorkSize[1]);

            kernel->setArg(argIdx++, openCLImage(input));
            kernel->setArg(argIdx++, openCLImage(output));

            auto runtime                    = mOpenCLBackend->getOpenCLRuntime();

            mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime());
            return NO_ERROR; 
        }

        ErrorCode ConvertExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
            MNN_PRINT("Start ConvertExecution onExecute... \n");
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
            MNN_PRINT("End ConvertExecution onExecute... \n");
#endif
            return NO_ERROR;
        }

        OpenCLCreatorRegister<TypedCreator<ConvertExecution>> __ConvertExecution(OpType_ConvertTensor);
        OpenCLCreatorRegister<TypedCreator<ConvertExecution>> __SqueezeExecution(OpType_Squeeze);

    } // namespace OpenCL
} // namespace MNN
