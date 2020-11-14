#include "MultiPipeline.hpp"
#include "Pipeline.hpp"
#include <memory>

namespace MNN {
MultiPipeline::MultiPipeline(std::vector<Pipeline *> pipelines)
    :mPipelines(pipelines) {}

ErrorCode MultiPipeline::prepare() {
    for (auto pipeline : mPipelines) {
        auto code = pipeline->prepare();
        MNN_ASSERT(code == NO_ERROR);
        if (code != NO_ERROR) {
            return code;
        }
    }

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
            mMultiUnits.push_back(std::make_shared<MultiUnit>(multiUnits));
        }
        unitIndex++;
    } while(!multiUnits.empty());


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


MultiPipeline::MultiUnit::MultiUnit(std::vector<std::vector<Unit*>> units)
    :mUnits(units) {
}

ErrorCode MultiPipeline::MultiUnit::prepare() {
    // TODO : merge ops & prepare kernel
}

ErrorCode MultiPipeline::MultiUnit::execute() {
    for (auto units : mUnits) {
        for (auto unit : units) {
            auto code = unit->execute();
            MNN_ASSERT(code == NO_ERROR);
            if (code != NO_ERROR) {
                return code;
            }
        }
    }
    return NO_ERROR;
}
}