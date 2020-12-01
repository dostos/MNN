__kernel void spatial_product(GLOBAL_SIZE_3_DIMS(0) __read_only image2d_t input, __read_only image2d_t input1,
                              __private const int out_height, __write_only image2d_t output) {
    const int channel_block_idx = get_global_id(0);
    const int w                 = get_global_id(1);
    const int hb                = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(0, 0, 0, 0, channel_block_idx, w, hb);
    const int width = global_size_dim1;

    const int h     = hb % out_height;
    const int pos   = mad24(channel_block_idx, width, w);
    FLOAT4 in_0 = RI_F(input, SAMPLER, (int2)(pos, hb));
    FLOAT4 in_1 = RI_F(input1, SAMPLER, (int2)(w, h));
    FLOAT4 out  = in_0 * in_1.x;
    WI_F(output, (int2)(pos, hb), out);
}
