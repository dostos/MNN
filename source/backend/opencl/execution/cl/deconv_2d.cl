__kernel void deconv_2d(GLOBAL_SIZE_3_DIMS __read_only image2d_t input, __read_only image2d_t weights,
#ifdef BIAS
                        __read_only image2d_t bias,
#endif
                        __write_only image2d_t output,
                        __private const int2 input_shape,
                        __private const int2 output_shape,
                        __private const int2 stride_shape,
                        __private const int2 align_shape,
                        __private const int2 padding_shape, 
                        __private const int2 kernel_shape,
                        __private const int kernel_size,
                        __private const int in_channel_blocks, __private const int out_channel_blocks) {

    const int out_channel_blocks_idx = get_global_id(0);
    const int out_w_idx          = get_global_id(1);
    const int out_batch_height_idx   = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(out_channel_blocks_idx, out_w_idx, out_batch_height_idx);

#ifdef BIAS
    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_blocks_idx, 0));
#else
    FLOAT4 out0 = (FLOAT4)0;
#endif

    const int out_b_idx  = out_batch_height_idx / output_shape.x;
    const int out_h_idx = out_batch_height_idx % output_shape.x;

    int kernel_start_x = max(0, (out_w_idx + align_shape.y) / stride_shape.y);
    int kernel_start_y = max(0, (out_h_idx + align_shape.x) / stride_shape.x);

    int deal_kernel_width  = kernel_shape.y - mad24(kernel_start_x, stride_shape.y, padding_shape.y) + out_w_idx - 1;
    int deal_kernel_height = kernel_shape.x - mad24(kernel_start_y, stride_shape.x, padding_shape.x) + out_h_idx - 1;

    int kernel_x_0, kernel_x_1, kernel_x_2, kernel_x_3, kernel_y;
    FLOAT4 in0;
    FLOAT4 weights0, weights1, weights2, weights3;
    for (int ic = 0; ic < in_channel_blocks; ic++) {
        kernel_x_0 = ic << 2;
        kernel_x_1 = kernel_x_0 + 1;
        kernel_x_2 = kernel_x_0 + 2;
        kernel_x_3 = kernel_x_0 + 3;
        for (int k_y = deal_kernel_height, idx_h = kernel_start_y; k_y >= 0; k_y -= stride_shape.x, idx_h++) {
            int in_idy      = mad24(out_b_idx, input_shape.x, idx_h);
            int in_hb_value = select(in_idy, -1, idx_h < 0 || idx_h >= input_shape.x);
            int in_width0   = kernel_start_x;
            for (int k_x = deal_kernel_width; k_x >= 0; k_x -= stride_shape.y) {
                kernel_y = mad24(k_y, kernel_shape.y, k_x);
                kernel_y = mad24(out_channel_blocks_idx, kernel_size, kernel_y);
                weights0 = RI_F(weights, SAMPLER, (int2)(kernel_x_0, kernel_y));
                weights1 = RI_F(weights, SAMPLER, (int2)(kernel_x_1, kernel_y));
                weights2 = RI_F(weights, SAMPLER, (int2)(kernel_x_2, kernel_y));
                weights3 = RI_F(weights, SAMPLER, (int2)(kernel_x_3, kernel_y));

                int in_idx = mul24(ic, input_shape.y);
                int in_width_value0 = in_width0;                                                           \
                in_width_value0 =                                                                                   \
                    select(in_idx + in_width_value0, -1, (in_width_value0 < 0 || in_width_value0 >= input_shape.y)); \
                in0 = RI_F(input, SAMPLER, (int2)(in_width_value0, in_hb_value));

                out0 = mad(in0.x, weights0, out0);
                out0 = mad(in0.y, weights1, out0);
                out0 = mad(in0.z, weights2, out0);
                out0 = mad(in0.w, weights3, out0);
                in_width0++;
            }
        }
    }
#ifdef RELU
    out0 = fmax(out0, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
#endif

    int out_image_width_idx = mad24(out_channel_blocks_idx, output_shape.y, out_w_idx);
    WI_F(output, (int2)(out_image_width_idx, out_batch_height_idx), out0);
}

__kernel void iohw2oihw(__global const float* input_ptr, __global float* output_ptr, int plane_number, int input_channel, int output_channel) {
    const int ic_index = get_global_id(0), oc_index = get_global_id(1);
    if (ic_index >= input_channel || oc_index >= output_channel) {
        return;
    }
    const int input_offset = (ic_index * output_channel + oc_index) * plane_number;
    const int output_offset = (oc_index * input_channel + ic_index) * plane_number;
    for (int i = 0; i < plane_number; ++i) {
        output_ptr[output_offset + i] = input_ptr[input_offset + i];
    }
}
