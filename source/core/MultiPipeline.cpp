#include "Backend.hpp"
#include "MultiPipeline.hpp"
#include "MultiExecution.hpp"
#include "Pipeline.hpp"
#include <memory>

namespace MNN {
MultiPipeline::MultiPipeline(std::vector<Pipeline *> pipelines, Backend* backend)
    :mPipelines(pipelines), mBackend(backend) {}

ErrorCode MultiPipeline::prepare() {
    for (auto pipeline : mPipelines) {
        auto code = pipeline->prepare();
        MNN_ASSERT(code == NO_ERROR);
        if (code != NO_ERROR) {
            return code;
        }
    }

    // Fuse sub-pipeline into multi unit
    {
        // TODO(dostos)
        // Use unit's flops to rebuild units
        mMultiUnits.clear();

        int unitIndex = 0;
        std::vector<std::vector<Unit *>> multiUnits;
        do {
            multiUnits.clear();
            
            for(int pipelineIndex = 0 ; pipelineIndex < mPipelines.size(); pipelineIndex++) {
                std::vector<Unit*> units;
                if (unitIndex < mPipelines[pipelineIndex]->mUnits.size()) {
                    units.push_back(mPipelines[pipelineIndex]->mUnits[unitIndex].get());
                }
                
                if (!units.empty())
                    multiUnits.push_back(units);
            }

            if (!multiUnits.empty()) {
                mMultiUnits.push_back(std::make_shared<MultiUnit>(multiUnits, mBackend));
            }
            unitIndex++;
        } while(!multiUnits.empty());
    }

    // Prepare multi unit
    {
        for (auto multiUnit : mMultiUnits) {
            auto code = multiUnit->prepare();
            MNN_ASSERT(code == NO_ERROR);
            if (code != NO_ERROR) {
                return code;
            }
        }
    }

    return NO_ERROR;
}

ErrorCode MultiPipeline::run() {
    for (auto multiUnit : mMultiUnits) {
        auto code = multiUnit->execute();
        MNN_ASSERT(code == NO_ERROR);
        if (code != NO_ERROR) {
            return code;
        }

    }

    return NO_ERROR;
}

ErrorCode MultiPipeline::runWithCallBack(const TensorCallBackWithInfo &enterCallback, const TensorCallBackWithInfo &exitCallback) {
    for (auto multiUnit : mMultiUnits) {
        auto code = multiUnit->executeCallBack(enterCallback, exitCallback);
        MNN_ASSERT(code == NO_ERROR);
        if (code != NO_ERROR) {
            return code;
        }

    }

    return NO_ERROR;
}


MultiUnit::MultiUnit(std::vector<std::vector<Unit*>> units, Backend* backend)
    :mUnits(units), mBackend(backend), mMultiExecution(nullptr) {
    bool supportMultiExecution = true;
    std::string name;
    std::string type;
    float flops;
    // Fuse when there is multiple units
    if (mUnits.size() > 1) {
        std::vector<std::vector<Execution *>> multiExecutions;

        for (auto units : mUnits) {
            std::vector<std::vector<Tensor *>> subPipelineInput, subPipelineOutput;
            std::vector<Execution *> executions;
            for (auto unit : units) {
                subPipelineInput.push_back(unit->mInputs);
                subPipelineOutput.push_back(unit->mOutputs);
                executions.push_back(unit->mExecution.get());
                supportMultiExecution &= unit->mExecution->fusionable();
                if (!name.empty()) {
                    name += "+";
                    type += "+";
                }
                name += unit->name();
                type += unit->type();
                flops += unit->flops();
            }

            mInput.push_back(subPipelineInput);
            mOutput.push_back(subPipelineOutput);
            multiExecutions.push_back(executions);
        }

        if (supportMultiExecution) {
            auto creator = MultiExecution::getMultiExecutionCreator(backend->type());
            if (creator) {
                // merge ops & prepare kernel
                mMultiExecution = std::shared_ptr<MultiExecution>(creator->onCreate(multiExecutions, backend));
                mMultiExecution->mContent->name = name;
                mMultiExecution->mContent->type = type;
                mMultiExecution->mContent->flops = flops;
            }
        }
    }
}

ErrorCode MultiUnit::prepare() {
    // set kernel arguments
    if (mMultiExecution) {
        auto code = mMultiExecution->onPrepare(mInput, mOutput);
        MNN_ASSERT(code == NO_ERROR);
        if (code != NO_ERROR) {
            return code;
        }
    }
    return NO_ERROR;
}

ErrorCode MultiUnit::execute() {
    if (mMultiExecution) {
        mMultiExecution->onExecute();
    } else {
        for (auto units : mUnits) {
            for (auto unit : units) {
                auto code = unit->execute();
                MNN_ASSERT(code == NO_ERROR);
                if (code != NO_ERROR) {
                    return code;
                }
            }
        }
    }
    return NO_ERROR;

}
ErrorCode MultiUnit::executeCallBack(const TensorCallBackWithInfo &enterCallback, const TensorCallBackWithInfo &exitCallback) {
    if (mMultiExecution) {
        mMultiExecution->onExecuteCallback(enterCallback, exitCallback);
    } else {
        for (auto units : mUnits) {
            for (auto unit : units) {
                auto code = unit->executeCallBack(enterCallback, exitCallback);                
                MNN_ASSERT(code == NO_ERROR);
                if (code != NO_ERROR) {
                    return code;
                }
            }
        }
    }
    return NO_ERROR;
}
}