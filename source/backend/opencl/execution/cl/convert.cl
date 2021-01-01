__kernel void convert(GLOBAL_SIZE_2_DIMS(0)
                      __read_only image2d_t input, __write_only image2d_t output) {
    const int wc = get_global_id(0);
    const int hb  = get_global_id(1);
    SKIP_ID_2_DIMS(0, wc, hb);
    
    FLOAT4 in  = RI_F(input, SAMPLER, (int2)(wc, hb));
    FLOAT4 out = in;
    WI_F(output, (int2)(wc, hb), out);
}
