#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

#define UP_DIV(x, y) (((x)+(y)-1)/(y))

__kernel void image2col_1x1(GLOBAL_SIZE_2_DIMS
                                __read_only image2d_t input, 
                                __write_only image2d_t output,
                                __private const int in_c_4,
                                __private const int outputWidth,
                                __private const int outputHeight) {
//index : ib*ic/4, oh, ow
//input image ic/4, ih, iw * ic4
//output : temp image : (ib*oh*ow)/ 4, ic/4*(ib*oh*ow)%4*ic4

int image_width_idx  = get_global_id(0);
int image_height_idx = get_global_id(1);

DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);


}

__kernel void image2col(GLOBAL_SIZE_2_DIMS
                                __read_only image2d_t input, 
                                __write_only image2d_t output,
                                __private const int in_c_4,
                                __private const int outputWidth,
                                __private const int outputHeight) {

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