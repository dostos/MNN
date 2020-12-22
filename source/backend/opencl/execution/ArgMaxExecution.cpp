#include "backend/opencl/execution/ArgMaxExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {
ArgMaxExecution::ArgMaxExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : Execution(backend) {
}

ErrorCode ArgMaxExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    return NO_ERROR;
}

ErrorCode ArgMaxExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<ArgMaxExecution>> __argmax_op(OpType_ArgMax);
}
}