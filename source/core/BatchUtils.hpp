#ifndef BatchUtils_hpp
#define BatchUtils_hpp

#include <array>
#include <vector>
#include <MNN/MNNDefine.h>

namespace MNN {
    std::array<int32_t, 4> MNN_PUBLIC getBatchBits(const std::vector<int>& batchIndexes);
}  // namespace MNN
#endif // BatchUtils_hpp