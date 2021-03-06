//
//  BackendFactory.cpp
//  MNN
//
//  Created by MNN on 2018/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/BackendFactory.hpp"
//#include <MNN/core/CPUBackend.hpp
#include "core/Macro.h"
#include <map>

namespace MNN {
Backend* BackendFactory::create(const Backend::Info& info) {
    auto creator = MNNGetExtraBackendCreator(info.type);
    if (nullptr == creator) {
        MNN_PRINT("Create Backend Failed because no creator for %d\n", info.type);
        return nullptr;
    }

    std::map<MNNForwardType, Backend*> backends;

    if (backends.find(info.type) == backends.end()) {
        auto backend = creator->onCreate(info);
        if (nullptr == backend) {
            MNN_PRINT("Create Backend failed, the creator return nullptr, type = %d\n", info.type);
        }
        backends[info.type] = backend;
    }
    return backends[info.type];
}
} // namespace MNN
