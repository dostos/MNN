#ifndef MultiPipeline_hpp
#define MultiPipeline_hpp

#include <vector>
#include <memory>
#include <MNN/ErrorCode.hpp>

namespace MNN {
class Pipeline;
class Unit;

class MultiPipeline {
public:
    MultiPipeline(std::vector<Pipeline *> pipelines);

    ErrorCode prepare();
    ErrorCode run();

    class MultiUnit {
    public:
        MultiUnit(std::vector<std::vector<Unit*>> units);

        ErrorCode prepare();
        ErrorCode execute();
    private:
        // units that should run parallel 
        std::vector<std::vector<Unit*>> mUnits;
    };

private:
    // pipelines that should run parallel 
    std::vector<Pipeline *> mPipelines;
    std::vector<std::shared_ptr<MultiUnit>> mMultiUnits;
};
} // namespace MNN

#endif