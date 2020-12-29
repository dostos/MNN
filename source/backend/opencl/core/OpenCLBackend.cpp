//
//  OpenCLBackend.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/core/OpenCLBackend.hpp"
#include "MNN_generated.h"

#include "core/TensorUtils.hpp"
#include "core/SizeComputer.hpp"
#include <map>
#include <mutex>
#include <thread>
#include "core/Macro.h"

namespace MNN {
namespace OpenCL {

std::map<OpType, OpenCLBackend::Creator*>* gCreator() {
    static std::once_flag once;
    static std::map<OpType, OpenCLBackend::Creator*>* creators = nullptr;
    std::call_once(once, [&]() { creators = new std::map<OpType, OpenCLBackend::Creator*>; });
    return creators;
};

OpenCLBackend::OpenCLBackend(BackendConfig::PrecisionMode precision, BackendConfig::PowerMode power)
    : Backend(MNN_FORWARD_OPENCL) {
    mPrecision = precision;
    // Shader precision
    if (precision == BackendConfig::Precision_Low) {
        mOpenCLRuntime.reset(new OpenCLRuntime(true));
    } else {
        mOpenCLRuntime.reset(new OpenCLRuntime(false));
    }
    if(mOpenCLRuntime.get()){
        if(mOpenCLRuntime->isCreateError() == true){
            mIsCreateError = true;
        }
        // Mid memory precision
        cl_channel_type dataType = CL_HALF_FLOAT;
        if (precision == BackendConfig::Precision_High) {
            dataType = CL_FLOAT;
        }
        mImagePool.reset(new ImagePool(mOpenCLRuntime->context(), dataType));
        mStaticImagePool.reset(new ImagePool(mOpenCLRuntime->context(), dataType));
        mBufferPool.reset(new BufferPool(mOpenCLRuntime->context(), CL_MEM_READ_WRITE));
        mBufferPoolInt8.reset(new BufferPoolInt8(mOpenCLRuntime->context(), CL_MEM_READ_WRITE));
        std::set<std::string> buildOptions;
        mNC4HW4BufferToImageFloat = mOpenCLRuntime->buildKernel("buffer_to_image", "nc4hw4_buffer_to_image", buildOptions);
        mNCHWBufferToImageFloat = mOpenCLRuntime->buildKernel("buffer_to_image", "nchw_buffer_to_image", buildOptions);
        mNHWCBufferToImageFloat = mOpenCLRuntime->buildKernel("buffer_to_image", "nhwc_buffer_to_image", buildOptions);
        mImageToNC4HW4BufferFloat = mOpenCLRuntime->buildKernel("buffer_to_image", "image_to_nc4hw4_buffer", buildOptions);
        mImageToNHWCBufferFloat = mOpenCLRuntime->buildKernel("buffer_to_image", "image_to_nhwc_buffer", buildOptions);
        mImageToNCHWBufferFloat = mOpenCLRuntime->buildKernel("buffer_to_image", "image_to_nchw_buffer", buildOptions);
    }
}

OpenCLBackend::~OpenCLBackend() {
#ifdef LOG_VERBOSE
    MNN_PRINT("enter OpenCLBackend::~OpenCLBackend \n");
#endif
}

OpenCLRuntime* OpenCLBackend::getOpenCLRuntime() {
    return mOpenCLRuntime.get();
}

bool OpenCLBackend::onAcquireBuffer(const Tensor* nativeTensor, StorageType storageType) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start OpenCLBackend::onAcquireBuffer !\n");
#endif

    //int8
    if(nativeTensor->getType().code == halide_type_int && nativeTensor->getType().bits == 8){

        unsigned int size = nativeTensor->size();
#ifdef LOG_VERBOSE
    MNN_PRINT("enter int8 alloc ! size : %d \n", size);
#endif
        if (storageType == DYNAMIC_SEPERATE || storageType == STATIC) {
            auto buffer                               = mBufferPoolInt8->alloc(size, true);
            ((Tensor*)nativeTensor)->buffer().device = (uint64_t)buffer; // fix
            return true;
        }
        if (storageType == DYNAMIC) {
            auto buffer                               = mBufferPoolInt8->alloc(size);
            ((Tensor*)nativeTensor)->buffer().device = (uint64_t)buffer; // fix
            return true;
        }
        return false;
    }
    auto tensorShape = OpenCL::tensorShapeFormat(nativeTensor);

    int N = tensorShape.at(0);
    int H = tensorShape.at(1);
    int W = tensorShape.at(2);
    int C = tensorShape.at(3);

    size_t imageWidth  = (size_t)UP_DIV(C, 4) * W;
    size_t imageHeight = (size_t)N * H;

    const std::vector<size_t> requestShape{imageWidth, imageHeight};
#ifdef LOG_VERBOSE
    MNN_PRINT("OpenCLBackend::onAcquireBuffer: [%d, %d, %d, %d], [%d, %d]\n", N, H, W, C, (int)imageWidth,
              (int)imageHeight);
#endif

    if (storageType == DYNAMIC_SEPERATE) {
        auto image                               = mImagePool->alloc(imageWidth, imageHeight, true);
        ((Tensor*)nativeTensor)->buffer().device = (uint64_t)image; // fix
        return true;
    }
    if (storageType == DYNAMIC) {
        auto image                               = mImagePool->alloc(imageWidth, imageHeight);
        ((Tensor*)nativeTensor)->buffer().device = (uint64_t)image; // fix
        return true;
    }
    MNN_ASSERT(storageType == STATIC);
    auto image                               = mStaticImagePool->alloc(imageWidth, imageHeight);
    ((Tensor*)nativeTensor)->buffer().device = (uint64_t)image; // fix
    return true;
}

bool OpenCLBackend::onReleaseBuffer(const Tensor* nativeTensor, StorageType storageType) {
    if(nativeTensor->getType().code == halide_type_int && nativeTensor->getType().bits == 8){

        return true;
    }
    if (storageType == DYNAMIC_SEPERATE) {
        return true;
    }
    auto image = (cl::Image*)nativeTensor->deviceId();
    if (storageType == DYNAMIC) {
        mImagePool->recycle(image);
        return true;
    }
    if (storageType == STATIC) {
        mStaticImagePool->recycle(image, true);
    }
    return true;
}
bool OpenCLBackend::onAllocateBuffer() {
    return true;
}

bool OpenCLBackend::onClearBuffer() {
    mImagePool->clear();
    mBufferPool->clear();
    mBufferPoolInt8->clear();
    return true;
}
std::pair<float, bool> OpenCLBackend::onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) {
    auto creators = gCreator();
    auto iter      = creators->find(op->type());
    if (iter == creators->end()) {
        return std::make_pair(0.0f, false);
    }
    const float defaultScheduleTime = 0.05f;
    auto flops = SizeComputer::computeFlops(op, inputs, outputs);

    auto computeFlops = mOpenCLRuntime->flops();
    return std::make_pair(defaultScheduleTime + flops / 1024.0f / computeFlops * 1000.0f, true);
}
Execution* OpenCLBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                   const MNN::Op* op) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start OpenCLBackend::onCreate \n");
#endif
    auto creators = gCreator();
    auto iter      = creators->find(op->type());
#if 0
    bool res = false;
#define PERMIT(t) if (op->type() == t) res = true
    PERMIT(OpType_Convolution);
    PERMIT(OpType_Deconvolution);
    PERMIT(OpType_Pooling);
    PERMIT(OpType_ReLU);
    //PERMIT(OpType_Softmax);
    PERMIT(OpType_UnaryOp);
    //PERMIT(OpType_SoftmaxGrad);
    PERMIT(OpType_Conv2DBackPropFilter);
#undef PERMIT
    if (!res) {
        return nullptr;
    }
#endif
    if (iter == creators->end()) {
        if (nullptr != op->name()) {
            MNN_PRINT("Don't support type %s, %s\n", EnumNameOpType(op->type()), op->name()->c_str());
        } else {
            MNN_PRINT("Don't support type %s\n", EnumNameOpType(op->type()));
        }
        return NULL;
    }

    auto maxImageSize = mOpenCLRuntime->getMaxImage2DSize();
    bool valid        = true;
    for (auto t : inputs) {
        int imageHeight = t->batch() * t->height();
        int imageWidth  = t->width() * UP_DIV(t->channel(), 4);
        if (imageHeight > maxImageSize.at(0) || imageWidth > maxImageSize.at(1)) {
            valid = false;
            break;
        }
    }
    for (auto t : outputs) {
        int imageHeight = t->batch() * t->height();
        int imageWidth  = t->width() * UP_DIV(t->channel(), 4);
        if (imageHeight > maxImageSize.at(0) || imageWidth > maxImageSize.at(1)) {
            valid = false;
            break;
        }
    }
    if (!valid) {
        MNN_PRINT("beyond cl_image creat size! fallback to cpu backend\n");
        return NULL;
    }

    auto exe = iter->second->onCreate(inputs, outputs, op, this);
    if (NULL == exe) {
        if (nullptr != op->name()) {
            MNN_PRINT("The Creator Don't support type %d, %s\n", op->type(), op->name()->c_str());
        } else {
//            MNN_PRINT("The Creator Don't support type %s\n", EnumNameOpType(op->type()));
        }
        return NULL;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("End OpenCLBackend::onCreate \n");
#endif
    return exe;
}

void OpenCLBackend::onResizeBegin() {
#ifdef ENABLE_OPENCL_TIME_PROFILER
    mOpenCLRuntime->setCommandQueueProfileEnable();
#endif
}

void OpenCLBackend::onResizeEnd() {
#ifdef ENABLE_OPENCL_TIME_PROFILER
    mOpenCLRuntime->setCommandQueueProfileDisable();
#endif
}

void OpenCLBackend::onExecuteBegin() const {
    mOpenCLRuntime->mQueueCount = 0;
}

void OpenCLBackend::onExecuteEnd() const {
    mOpenCLRuntime->mQueueCount = 0;
}

bool OpenCLBackend::onWaitFinish() {
    int rc = mOpenCLRuntime.get()->commandQueue().finish();
    return rc == 0;
}

bool OpenCLBackend::isCreateError() const {
    return mIsCreateError;
}

std::shared_ptr<cl::Buffer> OpenCLBackend::_getHostBuffer(int length, size_t index) const {
    MNN_ASSERT(length > 0);
    if (mHostBuffers.find(index) == mHostBuffers.end() || length > mHostBuffers[index].first) {
        mHostBuffers[index] = std::pair<int, std::shared_ptr<cl::Buffer>>{length, new cl::Buffer(mOpenCLRuntime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, length)};
    }
    return mHostBuffers[index].second;
}

void OpenCLBackend::copyFromDeviceInt8(const Tensor* srcTensor, const Tensor* dstTensor) const{
        auto needSize = dstTensor->size();
        auto hostPtr = dstTensor->host<float>();
        cl_int error                = CL_SUCCESS;
        auto DeviceBuffer = (cl::Buffer*)srcTensor->deviceId();
        mOpenCLRuntime->commandQueue().enqueueReadBuffer(*DeviceBuffer, CL_TRUE, 0, needSize, hostPtr);
}

void OpenCLBackend::copyToDeviceInt8(const Tensor* srcTensor, const Tensor* dstTensor) const{
        auto needSize = srcTensor->size();
        auto hostPtr                = srcTensor->host<int8_t>();
        cl_int error                = CL_SUCCESS;
        auto DeviceBuffer = (cl::Buffer*)dstTensor->deviceId();
        mOpenCLRuntime->commandQueue().enqueueWriteBuffer(*DeviceBuffer, CL_TRUE, 0, needSize, hostPtr);
}

void OpenCLBackend::copyFromDevice(const Tensor* srcTensor, const Tensor* dstTensor) const{
    std::vector<int> bufferShape = MNN::OpenCL::tensorShapeFormat(srcTensor);
    MNN::Tensor interBuffer(0, Tensor::TENSORFLOW);
    interBuffer.buffer().dimensions = bufferShape.size();
    for (int i = 0; i < bufferShape.size(); i++) {
        interBuffer.buffer().dim[i].extent = bufferShape.at(i);
    }
    auto needSize = dstTensor->size();
    auto hostTensor = _getHostBuffer(needSize);
    interBuffer.buffer().device = (uint64_t)hostTensor.get();

    MNN_DATA_FORMAT data_format = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
    switch (data_format) {
        case MNN_DATA_FORMAT_NHWC:
            OpenCL::convertImageToNHWCBuffer(srcTensor, &interBuffer,
                                             *const_cast<cl::Kernel*>(&mImageToNHWCBufferFloat), mOpenCLRuntime.get());
            break;
        case MNN_DATA_FORMAT_NCHW:
            OpenCL::convertImageToNCHWBuffer(srcTensor, &interBuffer,
                                             *const_cast<cl::Kernel*>(&mImageToNCHWBufferFloat), mOpenCLRuntime.get());
            break;
        case MNN_DATA_FORMAT_NC4HW4:
            OpenCL::convertImageToNC4HW4Buffer(
                srcTensor, &interBuffer, *const_cast<cl::Kernel*>(&mImageToNC4HW4BufferFloat), mOpenCLRuntime.get());
            break;
        default:
            break;
    }
    auto hostPtr = dstTensor->host<float>();
    cl_int error                = CL_SUCCESS;

    mOpenCLRuntime->commandQueue().enqueueReadBuffer(*hostTensor, CL_TRUE, 0, needSize, hostPtr);
}


void OpenCLBackend::copyFromDevices(const std::vector<std::pair<Tensor*, Tensor*>>& tensors) const {
    std::vector<std::shared_ptr<MNN::Tensor>> interBuffers;
    std::vector<std::pair<Tensor *, Tensor *>> nhwc, nchw, nc4hw4;
    for (int i = 0; i < tensors.size(); i++) {
        interBuffers.push_back(std::make_shared<MNN::Tensor>(0, Tensor::TENSORFLOW));
        std::vector<int> bufferShape = MNN::OpenCL::tensorShapeFormat(tensors[i].first);
        interBuffers[i]->buffer().dimensions = bufferShape.size();
        for (int j = 0; j < bufferShape.size(); j++) {
            interBuffers[i]->buffer().dim[j].extent = bufferShape.at(j);
        }
        auto needSize = tensors[i].second->size();
        auto hostTensor = _getHostBuffer(needSize, i);
        interBuffers[i]->buffer().device = (uint64_t)hostTensor.get();

        MNN_DATA_FORMAT data_format = TensorUtils::getDescribe(tensors[i].second)->dimensionFormat;
        switch (data_format) {
            case MNN_DATA_FORMAT_NHWC:
                nhwc.push_back({tensors[i].first, interBuffers[i].get()});
                break;
            case MNN_DATA_FORMAT_NCHW:
                nchw.push_back({tensors[i].first, interBuffers[i].get()});
                break;
            case MNN_DATA_FORMAT_NC4HW4:
                nc4hw4.push_back({tensors[i].first, interBuffers[i].get()});
                break;
            default:
                break;
        }
    }

    if (!nhwc.empty() || !nchw.empty()) {
        for (auto& tensor : nhwc) {
            OpenCL::convertImageToNHWCBuffer(tensor.first, tensor.second,
                                             *const_cast<cl::Kernel*>(&mImageToNHWCBufferFloat), mOpenCLRuntime.get());
        }
        for (auto& tensor : nchw) {
            OpenCL::convertImageToNCHWBuffer(tensor.first, tensor.second,
                                             *const_cast<cl::Kernel*>(&mImageToNCHWBufferFloat), mOpenCLRuntime.get());
        }
    }

    if (!nc4hw4.empty()) {
        std::string kernelKey = "image_to_nc4hw4_buffer" + std::to_string(nc4hw4.size());
        OpenCL::convertImageToNC4HW4Buffers(nc4hw4, *const_cast<cl::Kernel*>(&mImageKernels[kernelKey]), mOpenCLRuntime.get());
    }
    
    for (int i = 0; i < tensors.size(); i++) {
        auto hostPtr = tensors[i].second->host<float>();
        auto needSize = tensors[i].second->size();
        auto hostTensor = _getHostBuffer(needSize, i);
        cl_int error                = CL_SUCCESS;

        mOpenCLRuntime->commandQueue().enqueueReadBuffer(*hostTensor, i == tensors.size() -1 ? CL_TRUE : CL_FALSE, 0, needSize, hostPtr);
    }
}

void OpenCLBackend::copyToDevice(const Tensor* srcTensor, const Tensor* dstTensor) const{
    bool needWait = false;
#ifdef ENABLE_OPENCL_TIME_PROFILER
    needWait = true;
#endif
    std::vector<int> bufferShape = MNN::OpenCL::tensorShapeFormat(srcTensor);
    MNN::Tensor interBuffer(0, Tensor::TENSORFLOW);
    interBuffer.buffer().dimensions = bufferShape.size();
    for (int i = 0; i < bufferShape.size(); i++) {
        interBuffer.buffer().dim[i].extent = bufferShape.at(i);
    }
    auto needSize = srcTensor->size();
    auto hostTensor = _getHostBuffer(needSize);
    interBuffer.buffer().device = (uint64_t)hostTensor.get();
    auto hostPtr                = srcTensor->host<float>();
    cl_int error                = CL_SUCCESS;
    mOpenCLRuntime->commandQueue().enqueueWriteBuffer(*hostTensor, CL_FALSE, 0, needSize, hostPtr);
    // Host -> OpenCL
    MNN_DATA_FORMAT data_format = TensorUtils::getDescribe(srcTensor)->dimensionFormat;
    if (MNN_DATA_FORMAT_NHWC == data_format) {
        OpenCL::convertNHWCBufferToImage(&interBuffer, const_cast<Tensor*>(dstTensor),
                                         *const_cast<cl::Kernel*>(&mNHWCBufferToImageFloat), mOpenCLRuntime.get(), needWait);
        return;
    }
    if (MNN_DATA_FORMAT_NCHW == data_format) {
        OpenCL::convertNCHWBufferToImage(&interBuffer, const_cast<Tensor*>(dstTensor),
                                         *const_cast<cl::Kernel*>(&mNCHWBufferToImageFloat), mOpenCLRuntime.get(), needWait);
        return;
    }
    if (MNN_DATA_FORMAT_NC4HW4 == data_format) {
        OpenCL::convertNC4HW4BufferToImage(&interBuffer, const_cast<Tensor*>(dstTensor),
                                           *const_cast<cl::Kernel*>(&mNC4HW4BufferToImageFloat),
                                           mOpenCLRuntime.get(), needWait);
        return;
    }
    MNN_ASSERT(false);
    return;
}

void OpenCLBackend::copyToDevices(const std::vector<std::pair<Tensor*, Tensor*>>& tensors) const {
    bool needWait = false;
#ifdef ENABLE_OPENCL_TIME_PROFILER
    needWait = true;
#endif
    
    std::vector<std::shared_ptr<MNN::Tensor>> interBuffers;
    std::vector<std::pair<Tensor *, Tensor *>> nhwc, nchw, nc4hw4;
    for (int i = 0; i < tensors.size(); i++) {
        std::vector<int> bufferShape = MNN::OpenCL::tensorShapeFormat(tensors[i].first);
        interBuffers.push_back(std::make_shared<MNN::Tensor>(0, Tensor::TENSORFLOW));
        interBuffers[i]->buffer().dimensions = bufferShape.size();
        for (int j = 0; j < bufferShape.size(); j++) {
            interBuffers[i]->buffer().dim[j].extent = bufferShape.at(j);
        }
        auto needSize = tensors[i].first->size();
        auto hostTensor = _getHostBuffer(needSize, i);
        interBuffers[i]->buffer().device = (uint64_t)hostTensor.get();
        auto hostPtr                = tensors[i].first->host<float>();
        cl_int error                = CL_SUCCESS;
        mOpenCLRuntime->commandQueue().enqueueWriteBuffer(*hostTensor, CL_FALSE, 0, needSize, hostPtr);
        
        // Host -> OpenCL
        MNN_DATA_FORMAT data_format = TensorUtils::getDescribe(tensors[i].first)->dimensionFormat;
        if (MNN_DATA_FORMAT_NHWC == data_format) {
            OpenCL::convertNHWCBufferToImage(interBuffers[i].get(), const_cast<Tensor*>(tensors[i].second),
                                            *const_cast<cl::Kernel*>(&mNHWCBufferToImageFloat), mOpenCLRuntime.get(), needWait);
        }
        else if (MNN_DATA_FORMAT_NCHW == data_format)
        {
            OpenCL::convertNCHWBufferToImage(interBuffers[i].get(), const_cast<Tensor*>(tensors[i].second),
                                            *const_cast<cl::Kernel*>(&mNCHWBufferToImageFloat), mOpenCLRuntime.get(), needWait);
        }
        else if (MNN_DATA_FORMAT_NC4HW4 == data_format) {
            nc4hw4.push_back({interBuffers[i].get(), tensors[i].second});
        }
    }
    

    if (!nc4hw4.empty()) {
        std::string kernelKey = "nc4hw4_buffer_to_image" + std::to_string(nc4hw4.size());
        OpenCL::convertNC4HW4BufferToImages(nc4hw4, *const_cast<cl::Kernel*>(&mImageKernels[kernelKey]), mOpenCLRuntime.get(), needWait);
    }
}

void OpenCLBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start onCopyBuffer !\n");
#endif
    //int8
    if(srcTensor->getType().code == halide_type_int && srcTensor->getType().bits == 8){
        if (srcTensor->deviceId() == 0 && dstTensor->deviceId() != 0) {
            copyToDeviceInt8(srcTensor, dstTensor);
        }else if(srcTensor->deviceId() != 0 && dstTensor->deviceId() == 0){
            copyFromDeviceInt8(srcTensor, dstTensor);
        }else{
            MNN_PRINT("onCopyBuffer int8 error !!! \n");
        }
    }else{
        if (srcTensor->deviceId() == 0 && dstTensor->deviceId() != 0) {
            copyToDevice(srcTensor, dstTensor);
        }else if(srcTensor->deviceId() != 0 && dstTensor->deviceId() == 0){
            copyFromDevice(srcTensor, dstTensor);
        }else{
            MNN_PRINT("onCopyBuffer float error !!! \n");
        }
    }

#ifdef LOG_VERBOSE
    MNN_PRINT("end onCopyBuffer !\n");
#endif
}

void OpenCLBackend::onCopyBuffers(const std::vector<Tensor *> &srcTensors, const std::vector<Tensor *> &dstTensors) const {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start onCopyBuffer !\n");
#endif

    std::vector<std::pair<Tensor *, Tensor *>> ctdInt8, cfdInt8, ctd, cfd;
    for (int i = 0; i < srcTensors.size(); i++) {
        if (srcTensors[i]->getType().code == halide_type_int && srcTensors[i]->getType().bits == 8){
            if (srcTensors[i]->deviceId() == 0 && dstTensors[i]->deviceId() != 0) {
                ctdInt8.push_back({srcTensors[i], dstTensors[i]});
            } else if(srcTensors[i]->deviceId() != 0 && dstTensors[i]->deviceId() == 0){
                cfdInt8.push_back({srcTensors[i], dstTensors[i]});
            } else{
                MNN_PRINT("onCopyBuffer int8 error !!! \n");
            }
        } else {
            if (srcTensors[i]->deviceId() == 0 && dstTensors[i]->deviceId() != 0) {
                ctd.push_back({srcTensors[i], dstTensors[i]});
            } else if(srcTensors[i]->deviceId() != 0 && dstTensors[i]->deviceId() == 0){
                cfd.push_back({srcTensors[i], dstTensors[i]});
            } else{
                MNN_PRINT("onCopyBuffer float error !!! \n");
            }
        }
    }

    for (auto& tensors: ctdInt8) {
        copyToDeviceInt8(tensors.first, tensors.second);
    }

    for (auto& tensors: cfdInt8) {
        copyFromDeviceInt8(tensors.first, tensors.second);
    }

    if (!ctd.empty()) {
        if (ctd.size() == 1) {
            copyToDevice(ctd.front().first, ctd.front().second);
        }
        else
        {
            copyToDevices(ctd);
        }
    }

    if (!cfd.empty()) {
        if (cfd.size() == 1) {
            copyFromDevice(cfd.front().first, cfd.front().second);
        }
        else
        {
            copyFromDevices(cfd);
        }
    }

#ifdef LOG_VERBOSE
    MNN_PRINT("end onCopyBuffer !\n");
#endif
}


bool OpenCLBackend::addCreator(OpType t, Creator* c) {
    auto map = gCreator();
    if (map->find(t) != map->end()) {
        MNN_PRINT("Error: %d type has be added\n", t);
        return false;
    }
    map->insert(std::make_pair(t, c));
    return true;
}

class CLBackendCreator : public BackendCreator {
public:
    virtual std::shared_ptr<Backend> onCreate(const Backend::Info& info) const override {
#ifdef MNN_USE_LIB_WRAPPER
        OpenCLSymbolsOperator::createOpenCLSymbolsOperatorSingleInstance();
        if (nullptr == OpenCLSymbolsOperator::getOpenclSymbolsPtr()) {
            MNN_PRINT("OpenCL init error , callback ... \n");
            return nullptr;
        }
        if (true == OpenCLSymbolsOperator::getOpenclSymbolsPtr()->isError()) {
            MNN_PRINT("parsing symbols error !!! \n");
            return nullptr;
        }
#endif
        BackendConfig::PrecisionMode precision = BackendConfig::Precision_Normal;
        BackendConfig::PowerMode power         = BackendConfig::Power_Normal;
        if (nullptr != info.user) {
            precision = info.user->precision;
            power     = info.user->power;
        }
        
        std::shared_ptr<OpenCLBackend> backend = std::shared_ptr<OpenCLBackend>(new OpenCLBackend(precision, power));
        
        if(!backend->isCreateError()){
            return backend;
        }else{
            return nullptr;
        }
    }
};

static const auto __opencl_global_initializer = []() {
    MNNInsertExtraBackendCreator(MNN_FORWARD_OPENCL, new CLBackendCreator, true);
    return true;
}();
} // namespace OpenCL
} // namespace MNN
