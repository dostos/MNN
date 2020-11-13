#ifndef MultiPipeline_hpp
#define MultiPipeline_hpp

#include <vector>
#include <MNN/ErrorCode.hpp>

namespace MNN {
class Pipeline;

// Set of pipelines that should run parallel 
class MultiPipeline {
public:
    MultiPipeline(std::vector<Pipeline *> pipelines);

    ErrorCode prepare();
    ErrorCode run();

private:
    std::vector<Pipeline *> mPipelines;
};
} // namespace MNN

#endif