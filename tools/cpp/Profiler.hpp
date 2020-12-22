//
//  Profiler.hpp
//  MNN
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Profiler_hpp
#define Profiler_hpp

#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

namespace MNN {

/** Profiler for Ops */
class Profiler {
public:
    /**
     * @brief get shared instance.
     */
    static Profiler* getInstance();
    /**
    * @brief start profiler with op, name and inout tensors.
    * @param op        given op.
    */
    void start(const OperatorInfo* info);
    /**
     * @brief end profiler with op name and type.
     * @param name      op name.
     */
    void end(const OperatorInfo* info);
    /**
     * print profiler time result, grouped by type and sorter by time cost.
     * @param loops     loop count.
     */
    void printTimeByType(int loops = 1);
    /**
     * print profiler time result, grouped and sorter by op name.
     * @param loops     loop count.
     */
    void printTimeByName(int loops = 1);
    void printTimeByOrder(int loops = 1);
    void clear() {
        mMapByType.clear();
        mMapByName.clear();
        mVectorByOrder.clear();
        mMapNameOrder.clear();
        mTotalTime = 0.0f;
        mTotalMFlops = 0.0f;
    }

private:
    ~Profiler() = default;

private:
    struct Record {
        std::string name;
        std::string type;
        int64_t order;
        int64_t calledTimes;
        float costTime;
        float flops;
    };

    static Profiler* gInstance;
    uint64_t mStartTime = 0;
    uint64_t mEndTime   = 0;
    float mTotalTime    = 0.0f;
    float mTotalMFlops  = 0.0f;
    std::map<std::string, Record> mMapByType;
    std::map<std::string, Record> mMapByName;

    std::vector<Record> mVectorByOrder;
    std::map<std::string, uint32_t> mMapNameOrder;

private:
    Record& getTypedRecord(const OperatorInfo* info);
    Record& getNamedRecord(const OperatorInfo* info);
    Record& getOrderedRecord(const OperatorInfo* info);
};

} // namespace MNN

#endif /* Profiler_hpp */
