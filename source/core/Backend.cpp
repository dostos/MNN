//
//  Backend.cpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <mutex>
#include "MNN_generated.h"
#include "core/Macro.h"
#include "core/Backend.hpp"

namespace MNN {

void registerBackend();

static std::map<MNNForwardType, std::pair<const BackendCreator*, bool>>& GetExtraCreator() {
    static std::once_flag flag;
    static std::map<MNNForwardType, std::pair<const BackendCreator*, bool>>* gExtraCreator;
    std::call_once(flag,
                   [&]() { gExtraCreator = new std::map<MNNForwardType, std::pair<const BackendCreator*, bool>>; });
    return *gExtraCreator;
}

void Backend::onCopyBuffers(const std::vector<Tensor *> &srcTensors, const std::vector<Tensor *> &dstTensors) const {
    MNN_ASSERT(srcTensors.size() == dstTensors.size());

    for (int i = 0; i < srcTensors.size(); i++) {
        onCopyBuffer(srcTensors[i], dstTensors[i]);
    }
}

const BackendCreator* MNNGetExtraBackendCreator(MNNForwardType type) {
    registerBackend();

    auto& gExtraCreator = GetExtraCreator();
    auto iter           = gExtraCreator.find(type);
    if (iter == gExtraCreator.end()) {
        return nullptr;
    }
    return iter->second.first;;
}

bool MNNInsertExtraBackendCreator(MNNForwardType type, const BackendCreator* creator, bool needCheck) {
    auto& gExtraCreator = GetExtraCreator();
    if (gExtraCreator.find(type) != gExtraCreator.end()) {
        MNN_ASSERT(false && "duplicate type");
        return false;
    }
    gExtraCreator.insert(std::make_pair(type, std::make_pair(creator, needCheck)));
    return true;
}
} // namespace MNN
