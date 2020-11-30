#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define READ_INPUT_IMAGE(i, base)                                                                         \
    int in_width_value##i = in_width##i + base;                                                           \
    in_width_value##i =                                                                                   \
        select(in_idx + in_width_value##i, -1, (in_width_value##i < 0 || in_width_value##i >= input_shape.y)); \
    in##i = RI_F(input, SAMPLER, (int2)(in_width_value##i, in_hb_value));

#define CALCULATE_OUTPUT(i)                  \
    out##i = mad(in##i.x, weights0, out##i); \
    out##i = mad(in##i.y, weights1, out##i); \
    out##i = mad(in##i.z, weights2, out##i); \
    out##i = mad(in##i.w, weights3, out##i);    

#define CALCULATE_OUTPUT_OPT(i)                  \
    out##i = mad(in_sm##i[local_idx].x, weights0, out##i); \
    out##i = mad(in_sm##i[local_idx].y, weights1, out##i); \
    out##i = mad(in_sm##i[local_idx].z, weights2, out##i); \
    out##i = mad(in_sm##i[local_idx].w, weights3, out##i);   

#define DEAL_NON_UNIFORM_DIM2(i, input1, input2)                       \
    if (input1 >= global_size_dim0##i || input2 >= global_size_dim1##i) { \
        return;                                                     \
    }

#define DEAL_NON_UNIFORM_DIM3(i, input1, input2, input3)                                             \
    if (input1 >= global_size_dim0##i || input2 >= global_size_dim1##i || input3 >= global_size_dim2##i) { \
        return;                                                                                   \
    }

#define GLOBAL_SIZE_2_DIMS(i) __private const int global_size_dim0##i, __private const int global_size_dim1##i,

#define GLOBAL_SIZE_3_DIMS(i) \
    __private const int global_size_dim0##i, __private const int global_size_dim1##i, __private const int global_size_dim2##i,

#define UNIT 4

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\

// ROI Pooling
#define MIN_VALUE -FLT_MAX
