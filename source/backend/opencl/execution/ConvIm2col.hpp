#ifndef ConvIm2Col_hpp
#define ConvIm2Col_hpp

#include "core/Execution.hpp"
#include "backend/opencl/execution/ConvExecution.hpp"

#include <array>
#include <functional>
#include <memory>
#include <vector>
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
namespace MNN {
namespace OpenCL {

class ConvIm2ColExecution : public ConvCommonExecution {
public:
    ConvIm2ColExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~ConvIm2ColExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const Convolution2DCommon *mConv2dCommonParams;

    bool mIsConv1x1;
    int obxohxow_4;

    std::vector<uint32_t> mIm2colSize;
    std::vector<uint32_t> mGemmSize;
    std::vector<uint32_t> mCol2imSize;

    std::vector<uint32_t> mIm2colGlobalSize;
    std::vector<uint32_t> mGemmGlobalSize;
    std::vector<uint32_t> mCol2imGlobalSize;

    std::shared_ptr<cl::Image2D> mSrcTexture;
    std::shared_ptr<cl::Image2D> mDstTexture;

    cl::Kernel mIm2ColKernel;
    cl::Kernel mGemmKernel;
    cl::Kernel mCol2ImKernel;

    uint32_t mMaxWorkGroupSize;
    bool mIsTurn = false;
    OpenCLBackend *mOpenCLBackend;
    bool mConv1x1Opt{false};
    bool mUseLocalMem{false};

    std::shared_ptr<cl::Buffer> mKernelBuffer;
    std::shared_ptr<cl::Image2D> mKernel;
    std::shared_ptr<cl::Buffer> mBiasBuffer;
};

} // namespace OpenCL
} // namespace MNN



#endif