#ifndef MultiExecution_hpp
#define MultiExecution_hpp

#include <MNN/ErrorCode.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>
#include "core/NonCopyable.hpp"
#include <memory>

namespace MNN {
class Execution;
class Backend;

typedef std::vector< // Per pipeline
        std::vector< // Per sub-pipeline
        std::vector< // Per unit
            Tensor *>>> MultiExecutionTensors;

class MNN_PUBLIC MultiExecution : public NonCopyable {
public:
    MultiExecution(std::vector<std::vector<Execution *>> executions, Backend* backend);
    virtual ~MultiExecution() = default;

    virtual ErrorCode onPrepare(const MultiExecutionTensors &inputs, const MultiExecutionTensors &outputs) = 0;
    virtual ErrorCode onExecute(const MultiExecutionTensors &inputs, const MultiExecutionTensors &outputs) = 0;

    class Creator : public NonCopyable {
    public:
        virtual ~Creator() = default;
        virtual MultiExecution *onCreate(std::vector<std::vector<Execution *>> executions, Backend* backend) const = 0;
    };

    static bool insertMultiExecutionCreator(MultiExecution::Creator* creator, MNNForwardType type);
    static const Creator *getMultiExecutionCreator(MNNForwardType type);
protected:
    std::vector<std::vector<Execution *>> mExecutions;
    Backend *mBackend;
};
};

#endif
