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
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const std::vector<int>& batchIndexes) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const Convolution2DCommon *mConv2dCommonParams;

    int mInputDepth;
    bool mIsConv1x1;

    std::vector<int> mStrides{1, 1};
    std::vector<int> mPaddings{0, 0};
    std::vector<int> mDilations{1, 1};
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};

    std::shared_ptr<Tensor> mFilter;
    cl::Kernel mKernel;
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



#endif