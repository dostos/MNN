#include "MultiPipeline.hpp"
#include "Pipeline.hpp"

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
    return NO_ERROR;
}

ErrorCode MultiPipeline::run() {
    for (auto pipeline : mPipelines) {
        auto code = pipeline->execute();
        MNN_ASSERT(code == NO_ERROR);
        if (code != NO_ERROR) {
            return code;
        }
    }
    return NO_ERROR;

}
}