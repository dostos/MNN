#ifndef MultiPipeline_hpp
#define MultiPipeline_hpp

#include <vector>
#include <memory>
#include <MNN/ErrorCode.hpp>

#include "MultiExecution.hpp"

namespace MNN {
class Backend;
class Pipeline;
class Unit;
class MultiUnit;

class MultiPipeline
{
public:
    MultiPipeline(std::vector<Pipeline *> pipelines, Backend *backend);

    ErrorCode prepare();
    ErrorCode run();

private:
    Backend *mBackend;
    // pipelines that should run parallel
    std::vector<Pipeline *> mPipelines;
    std::vector<std::shared_ptr<MultiUnit>> mMultiUnits;
};

class MultiUnit {
public:
    MultiUnit(std::vector<std::vector<Unit*>> units, Backend *backend);

    ErrorCode prepare();
    ErrorCode execute();
private:
    Backend *mBackend;
    // units that should run parallel 
    std::vector<std::vector<Unit*>> mUnits;
    std::shared_ptr<MultiExecution> mMultiExecution;
    MultiExecutionTensors mInput, mOutput;
};
} // namespace MNN

#endif