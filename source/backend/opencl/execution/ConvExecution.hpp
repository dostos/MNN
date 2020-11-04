//
//  ConvExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvExecution_hpp
#define ConvExecution_hpp

#include "core/Execution.hpp"

#include <array>
#include <functional>
#include <memory>
#include <vector>
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
namespace MNN {
namespace OpenCL {

class ConvCommonExecution : public Execution {
public:
    ConvCommonExecution(const Convolution2D *conv_op, const MNN::Op *op, Backend *backend);
    virtual ~ConvCommonExecution();

protected:
    std::string mName;
    std::shared_ptr<Tensor> mBias;
};

class ConvExecution : public ConvCommonExecution {
public:
    ConvExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~ConvExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    static std::shared_ptr<Tensor> getBias(OpenCLBackend *backend, const Convolution2D *conv);

    std::vector<uint32_t> conv2d1x1LocalWS(std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize);
    std::vector<uint32_t> conv2d1x1LocalWSOpt(std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize);

    std::vector<uint32_t> conv2dGeneralLocalWS(const std::vector<uint32_t> &gws, const uint32_t kernelSize,
                                               const uint32_t maxWorkGroupSize);

private:
    const Convolution2DCommon *mConv2dCommonParams;
    std::vector<int> mStrides{1, 1};
    std::vector<int> mPaddings{0, 0};
    std::vector<int> mDilations{1, 1};
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
    std::shared_ptr<Tensor> mFilter;
    cl::Kernel mKernel;
    std::string mKernelName;
    uint32_t mMaxWorkGroupSize;
    bool mIsTurn = false;
    OpenCLBackend *mOpenCLBackend;
    bool mConv1x1Opt{false};
    bool mUseLocalMem{false};
    std::shared_ptr<cl::Buffer> mKernelBuffer;
    std::shared_ptr<cl::Buffer> mBiasBuffer;
};

} // namespace OpenCL
} // namespace MNN
#endif /* ConvExecution_hpp */
