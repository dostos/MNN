// get index'th 1 in bits
int getBatchIndex(__private const int4 batchIndexes, int index) {
    int output = 0;
    #pragma unroll
    for(int i = 0; i < 32; i++) {
        // Count # of 1 from left
        batchIndexes.x >> i & 1 ? index-- : 0; 
        // Assign index'th 1 in output & decrement index to discard the rest loop
        index == -1 ? index-- & (output = i) : 0; 
    }
    #pragma unroll
    for(int i = 0; i < 32; i++) {
        batchIndexes.y >> i & 1 ? index-- : 0;
        index == -1 ? index-- & (output = i + 32)  : 0;
    }
    #pragma unroll
    for(int i = 0; i < 32; i++) {
        batchIndexes.z >> i & 1 ? index-- : 0;
        index == -1 ? index-- & (output = i + 64) : 0;
    }
    #pragma unroll
    for(int i = 0; i < 32; i++) {
        batchIndexes.w >> i & 1 ? index-- : 0;
        index == -1 ? index-- & (output = i + 96) : 0;
    }
    return output;
}
