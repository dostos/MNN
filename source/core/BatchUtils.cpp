#include "BatchUtils.hpp"
#include <MNN/MNNDefine.h>

namespace MNN {

    std::array<int32_t, 4> getBatchBits(const std::vector<int>& batchIndexes) {
        std::array<int32_t, 4> batchBits = {0, 0, 0, 0};

        for(auto i : batchIndexes) {
            // TODO : Only support upto 128 (32 bit * 4)
            MNN_ASSERT(i < 128);
            batchBits.data()[i / 32] |= 1 << (i % 32);
        }

        return batchBits;
    }
}
