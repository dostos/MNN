#include "MultiExecution.hpp"
#include "Execution.hpp"
#include <MNN/MNNForwardType.h>
#include <map>
#include <mutex>

namespace MNN {

MultiExecution::MultiExecution(std::vector<std::vector<Execution *>> executions, Backend* backend)
    :mExecutions(executions), mBackend(backend) {}
    
static std::map<MNNForwardType, MultiExecution::Creator*>& getMultiExecutionCreators() {
    static std::once_flag flag;
    static std::map<MNNForwardType, MultiExecution::Creator*>* gExtraCreator;
    if (gExtraCreator == nullptr) {
        gExtraCreator = new std::map<MNNForwardType, MultiExecution::Creator*>;
    }
    return *gExtraCreator;
}

bool MultiExecution::insertMultiExecutionCreator(MultiExecution::Creator* creator, MNNForwardType type) {
    auto& gExtraCreator = getMultiExecutionCreators();
    if (gExtraCreator.find(type) != gExtraCreator.end()) {
        MNN_ASSERT(false && "duplicate type");
        return false;
    }
    gExtraCreator[type] = creator;
    return true;
}

const MultiExecution::Creator* MultiExecution::getMultiExecutionCreator(MNNForwardType type) {
    auto& gExtraCreator = getMultiExecutionCreators();
    auto iter           = gExtraCreator.find(type);
    if (iter == gExtraCreator.end()) {
        return nullptr;
    }
    return iter->second;
}
}