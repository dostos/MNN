#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

#define UP_DIV(x, y) (((x)+(y)-1)/(y))

__kernel void image2col_1x1(GLOBAL_SIZE_3_DIMS
                                __read_only image2d_t input, 
                                __write_only image2d_t output,
                                __private const int ic_4,
                                __private const int inputWidth,
                                __private const int inputHeight,
                                __private const int outputWidth,
                                __private const int outputHeight) {
    // w, h, ic4_b 
    int3 index = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

    DEAL_NON_UNIFORM_DIM3(index.x, index.y, index.z);

    int ic_4_i = index.z % ic_4;
    int ib_i = index.z / ic_4;
    int destYOrigin = ib_i*outputWidth*outputHeight + index.y*outputWidth + index.x;
    int destY = destYOrigin / 4;
    int destXOffset = destYOrigin % 4;
    float4 color = read_imagef(input, SAMPLER, (int2)(index.x + ic_4_i * inputWidth, index.y + ib_i * inputHeight));
    write_imagef(output, (int2)(ic_4_i*4+destXOffset, destY), color);
}

__kernel void image2col(GLOBAL_SIZE_3_DIMS
                                __read_only image2d_t input, 
                                __write_only image2d_t output,
                                __private const int2 pad,
                                __private const int2 kernelSize,
                                __private const int2 stride,
                                __private const int2 dilate,
                                __private const int4 inputSize,
                                __private const int4 outputSize) {
    // w, h, ic4_b 
    int3 index = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

    DEAL_NON_UNIFORM_DIM3(index.x, index.y, index.z);

    int2 s0 = index.xy*stride-pad;
    int2 sfxy = max((int2)(0), (UP_DIV(-s0, dilate)));
    int2 efxy = min(kernelSize, UP_DIV(inputSize.xy-s0, dilate));

    int ic_4 = index.z % inputSize.z; //input channel
    int ib = index.z / inputSize.z; // input batch

    int destYOrigin = ib*outputSize.x*outputSize.y + index.y*outputSize.x + index.x;
    int destY = destYOrigin / 4;
    int destXOffset = destYOrigin % 4;
    for (int fy=0; fy<kernelSize.y; ++fy)
    {
        int sy = fy*dilate.y + s0.y;
        for (int fx=0; fx<kernelSize.x; ++fx)
        {
            int sx = fx*dilate.x + s0.x;
            int destX = fx + fy*kernelSize.x + ic_4*kernelSize.x * kernelSize.y;
            float4 color = read_imagef(input, SAMPLER, (int2)(sx + ic_4 * inputSize.x, sy + ib * inputSize.y));
            write_imagef(output, (int2)(4*destX+destXOffset, destY), color);
        }
    }
}


__kernel void col2image(GLOBAL_SIZE_3_DIMS
                                __read_only image2d_t input, 
                                __write_only image2d_t output,
                                __global const float4 *bias,
                                __private const int4 outputSize) {
    // ow, oc, oc4_ob
    int3 index = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

    int ob = index.z / outputSize.z;
    int oc_4 = index.z % outputSize.z;
    
    if (index.x < outputSize.x && index.y < outputSize.y) {
        int sourceXIndex = ob*outputSize.x*outputSize.y + index.y*outputSize.x + index.x;
        int sourceX = sourceXIndex / 4;
        int sourceY = oc_4 * 4 + sourceXIndex % 4;
        float4 color = bias[oc_4];
        color += read_imagef(input, SAMPLER, (int2)(sourceX, sourceY));
#ifdef RELU
        color = max(color, float4(0));
#endif
#ifdef RELU6
        color = clamp(color, float4(0), float4(6));
#endif
        write_imagef(output, (int2)(index.x + oc_4 * outputSize.x, index.y + ob * outputSize.y), color);
    }
}

