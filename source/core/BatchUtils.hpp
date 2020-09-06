#ifndef BatchUtils_hpp
#define BatchUtils_hpp

#include <array>
#include <vector>

namespace MNN {
    std::array<int32_t, 4> getBatchBits(const std::vector<int>& batchIndexes);
}
#endif // BatchUtils_hpp