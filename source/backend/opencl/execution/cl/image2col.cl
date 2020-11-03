#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

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
                                __private const int in_c_4,
                                __private const int outputWidth,
                                __private const int outputHeight) {
// w, h, cb 
int3 pos = int3(get_global_id(0), get_global_id(1), get_global_id(2));

DEAL_NON_UNIFORM_DIM3(image_width_idx, image_height_idx);

int ic_4_i = pos.z % ic_4;
int ib_i = pos.z / ic_4;
int destYOrigin = ib_i*outputWidth*outputHeight + pos.y*outputWidth + pos.x;
int destY = destYOrigin / 4;
int destXOffset = destYOrigin % 4;
float4 color = read_imagef(input, int2(pos.x, pos.y, pos.z), 0);
write_imagef(output, int2(ic_4_i*4+destXOffset, destY), color);
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
int output_width_idx  = get_global_id(0);
int output_height_idx = get_global_id(1);
int output_channel_batch_idx = get_global_id(2);

DEAL_NON_UNIFORM_DIM3(image_width_idx, image_height_idx);

int2 s0 = index.xy*stride-pad;
int2 sfxy = max(ivec2(0), (UP_DIV(-s0, dilate)));
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
        float4 color = texelFetch(uInput, ivec3(sx, sy, index.z), 0);
        write_imagef(output, (int2)(4*destX+destXOffset, destY), color);
    }
}

}

//
//layout(std430) buffer;
//layout(binding=0, FORMAT) writeonly mediump uniform image2D uOutput;
//layout(location=1) uniform mediump sampler3D uInput;
//layout(location=5) uniform int ic_4;
//layout(location=6) uniform int outputWidth;
//layout(location=7) uniform int outputHeight;
//layout (local_size_x = XLOCAL, local_size_y = YLOCAL, local_size_z = ZLOCAL) in;
//#define UP_DIV(x, y) (((x)+(y)-1)/(y))
////index : ib*ic/4, oh, ow
////input image ic/4, ih, iw * ic4
////output : temp image : (ib*oh*ow)/ 4, ic/4*(ib*oh*ow)%4*ic4
//void main()
//{
//    ivec3 pos = ivec3(gl_GlobalInvocationID);
//    if (pos.x < outputWidth && pos.y < outputHeight)
//    {
//        int ic_4_i = pos.z % ic_4;
//        int ib_i = pos.z / ic_4;
//        int destYOrigin = ib_i*outputWidth*outputHeight + pos.y*outputWidth + pos.x;
//        int destY = destYOrigin / 4;
//        int destXOffset = destYOrigin % 4;
//        vec4 color = texelFetch(uInput, ivec3(pos.x, pos.y, pos.z), 0);
//        imageStore(uOutput, ivec2(ic_4_i*4+destXOffset, destY), color);
//    }
//}

//"layout(std430) buffer;
//"layout(binding=0, FORMAT) writeonly mediump uniform image2D uOutput;
//"layout(location=1) uniform mediump sampler3D uInput;
//"layout(location=2) uniform ivec2 pad;
//"layout(location=3) uniform ivec2 kernelSize;
//"layout(location=4) uniform ivec2 stride;
//"layout(location=5) uniform ivec2 dilate;
//"layout(location=6) uniform ivec4 inputSize;
//"layout(location=7) uniform ivec4 outputSize;
//"layout (local_size_x = XLOCAL, local_size_y = YLOCAL, local_size_z = ZLOCAL) in;
//"#define UP_DIV(x, y) (((x)+(y)-1)/(y))
//"//index : ib*ic/4, oh, ow
//"//input image ic/4, ih, iw * ic4
//"//inputsize : ic/4, ih, iw
//"//ouputsize : oc/4, oh, ow
//"//output : temp image : (ib*oh*ow)/ 4, ic/4*ky*kx*(ib*oh*ow)%4*ic4
//"void main()
//"{
//"    ivec3 index = ivec3(gl_GlobalInvocationID);
//"    if (index.x < outputSize.x && index.y < outputSize.y)
//"    {
//"        ivec2 s0 = index.xy*stride-pad;
//"        ivec2 sfxy = max(ivec2(0), (UP_DIV(-s0, dilate)));
//"        ivec2 efxy = min(kernelSize, UP_DIV(inputSize.xy-s0, dilate));
//"        int ic_4 = index.z % inputSize.z; //input channel
//"        int ib = index.z / inputSize.z; // input batch
//"        
//"        int destYOrigin = ib*outputSize.x*outputSize.y + index.y*outputSize.x + index.x;
//"        int destY = destYOrigin / 4;
//"        int destXOffset = destYOrigin % 4;
//"        for (int fy=0; fy<kernelSize.y; ++fy)
//"        {
//"            int sy = fy*dilate.y + s0.y;
//"            for (int fx=0; fx<kernelSize.x; ++fx)
//"            {
//"                int sx = fx*dilate.x + s0.x;
//"                int destX = fx + fy*kernelSize.x + ic_4*kernelSize.x * kernelSize.y;
//"                vec4 color = texelFetch(uInput, ivec3(sx, sy, index.z), 0);
//"                imageStore(uOutput, ivec2(4*destX+destXOffset, destY), color);
//"            }
//"        }
//"    }
//"}