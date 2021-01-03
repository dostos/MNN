//
//  OpenCLRuntime.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OpenCLRuntime_hpp
#define OpenCLRuntime_hpp


#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <sstream>
#include <string>
#include <vector>
#include "core/Macro.h"
#include "Type_generated.h"
#include "backend/opencl/core/runtime/OpenCLWrapper.hpp"
#include "backend/opencl/core/KernelCompiler.hpp"

namespace MNN {

#define CL_CONTEXT_PERF_HINT_QCOM 0x40C2
#define CL_PERF_HINT_HIGH_QCOM 0x40C3
#define CL_PERF_HINT_NORMAL_QCOM 0x40C4
#define CL_PERF_HINT_LOW_QCOM 0x40C5
#define CL_CONTEXT_PRIORITY_HINT_QCOM 0x40C9
#define CL_PRIORITY_HINT_HIGH_QCOM 0x40CA
#define CL_PRIORITY_HINT_NORMAL_QCOM 0x40CB
#define CL_PRIORITY_HINT_LOW_QCOM 0x40CC

#define CL_KERNEL_WAVE_SIZE_QCOM 0xAA02

enum GpuType { MALI = 0, ADRENO = 1, RADEON = 2, OTHER = 3 };

class OpenCLRuntime {
public:
    OpenCLRuntime(bool permitFloat16);
    ~OpenCLRuntime();
    OpenCLRuntime(const OpenCLRuntime &) = delete;
    OpenCLRuntime &operator=(const OpenCLRuntime &) = delete;

    bool isSupportedFP16() const;
    bool isSupportedDotInt8() const;
    bool isSupportedDotAccInt8() const;
    ::cl::Context &context();
    ::cl::CommandQueue &commandQueue();
    OpenCL::KernelCompiler &KernelCompiler();
    uint64_t deviceGlobalMemeryCacheSize() const;
    uint32_t deviceComputeUnits() const;
    uint32_t maxFreq() const;
    uint64_t getMaxWorkGroupSize(const ::cl::Kernel &kernel);
    uint64_t getKernelWaveSize(const cl::Kernel &kernel);
    uint64_t getKernelPreferredWorkGroupSize(const cl::Kernel &kernel);
    //uint64_t getMaxLocalMem() const;
    GpuType getGpuType();
    uint64_t maxAllocSize() const;
    void setCommandQueueProfileEnable();
    void setCommandQueueProfileDisable();
    
    unsigned int mQueueCount = 0;
    unsigned int getQueueNum();
    
    std::map<std::pair<std::string, std::vector<uint32_t>>, std::vector<uint32_t>>& tunedLwsMap();
    
    ::cl::Kernel buildKernel(const std::string &programName, const std::string &kernelName,
                             const std::set<std::string> &buildOptions);

                             
    ::cl::Kernel buildKernelFromSource(const std::string &kernelName, const std::string &source, 
                             const std::set<std::string> &buildOptions);

    std::string getProgramSource(const std::string &programName) const;

    std::vector<size_t> getMaxImage2DSize();
    bool isCreateError() const;

    float flops() const {
        return mFlops;
    }

    double getCostTime(const cl::Event *event);
    double getQueuedTime(const cl::Event *event);
    double getSubmitTime(const cl::Event *event);

private:
    bool loadProgram(const std::string &programName, cl::Program *program);
    bool buildProgram(const std::string &buildOptionsStr, cl::Program *program);
    bool getDeviceSupportsExtension(const cl::Device &device, const char *extensionName);

    std::string getCommonBuildString() const;

private:
    std::shared_ptr<::cl::Context> mContext;
    std::shared_ptr<::cl::Device> mFirstGPUDevicePtr;
    std::shared_ptr<::cl::CommandQueue> mCommandQueuePtr;
    std::shared_ptr<OpenCL::KernelCompiler> mKernelCompiler;
    std::map<std::string, ::cl::Program> mBuildProgramMap;
    std::map<std::string, ::cl::Kernel> mFusedKernelMap;
    uint64_t mGPUGlobalMemeryCacheSize;
    uint32_t mGPUComputeUnits;
    uint32_t mMaxFreq;
    uint32_t mMaxMemAllocSize;
    uint64_t mMaxLocalMemSize;
    bool mIsSupportedFP16     = false;
    bool mSupportDotInt8 = false;
    bool mSupportDotAccInt8 = false;
    GpuType mGpuType;
    bool mIsSetWorkGroupAttribute = true;
    std::string mDefaultBuildParams;
    float mFlops = 4.0f;
    bool mIsCreateError{false};
    
    double mStartNanos;
    double mStopNanos;

    static const char *sLwsCache;
    std::map<std::pair<std::string, std::vector<uint32_t>>, std::vector<uint32_t>> mTunedLws;

};

} // namespace MNN
#endif  /* OpenCLRuntime_hpp */
