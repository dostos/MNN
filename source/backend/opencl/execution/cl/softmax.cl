__kernel void softmax_channel(GLOBAL_SIZE_3_DIMS(0) 
                            __private const int channel_blocks,
                            __private const int output_width,
                            __read_only image2d_t input, 
                            __write_only image2d_t output, 
                            __private const int output_channels,
                            __private const int remain_channels) {

    const int channel_block_idx = get_global_id(0);
    const int width_idx    = get_global_id(1);
    const int batch_height_idx       = get_global_id(2);

    SKIP_ID_3_DIMS(0, channel_block_idx, width_idx, batch_height_idx);

    const int width     = output_width;

    FLOAT float_max_value = -FLT_MAX;
    FLOAT4 input_data;
    for (short i = 0; i < channel_blocks - 1; ++i) {
        input_data      = RI_F(input, SAMPLER, (int2)(width_idx + i * output_width, batch_height_idx));
        float_max_value = max(float_max_value, input_data.x);
        float_max_value = max(float_max_value, input_data.y);
        float_max_value = max(float_max_value, input_data.z);
        float_max_value = max(float_max_value, input_data.w);
    }

    input_data = RI_F(input, SAMPLER, (int2)(width_idx + (channel_blocks - 1) * output_width , batch_height_idx));
    if (remain_channels == 0) {
        float_max_value = max(float_max_value, input_data.w);
        float_max_value = max(float_max_value, input_data.z);
        float_max_value = max(float_max_value, input_data.y);
        float_max_value = max(float_max_value, input_data.x);
    } else if (remain_channels == 1) {
        float_max_value = max(float_max_value, input_data.z);
        float_max_value = max(float_max_value, input_data.y);
        float_max_value = max(float_max_value, input_data.x);
    } else if (remain_channels == 2) {
        float_max_value = max(float_max_value, input_data.y);
        float_max_value = max(float_max_value, input_data.x);
    } else if (remain_channels == 3) {
        float_max_value = max(float_max_value, input_data.x);
    }

    FLOAT accum_result       = 0;
    for (short i = 0; i < channel_blocks - 1; ++i) {
        input_data = RI_F(input, SAMPLER, (int2)(width_idx + i * output_width, batch_height_idx));
        input_data = exp(input_data - float_max_value);
        accum_result += input_data.x;
        accum_result += input_data.y;
        accum_result += input_data.z;
        accum_result += input_data.w;
    }

    input_data = RI_F(input, SAMPLER, (int2)(width_idx + (channel_blocks - 1) * output_width, batch_height_idx));
    input_data -= float_max_value;
    if (remain_channels == 0) {
        accum_result += exp(input_data.w);
        accum_result += exp(input_data.z);
        accum_result += exp(input_data.y);
        accum_result += exp(input_data.x);
    } else if (remain_channels == 1) {
        accum_result += exp(input_data.z);
        accum_result += exp(input_data.y);
        accum_result += exp(input_data.x);
    } else if (remain_channels == 2) {
        accum_result += exp(input_data.y);
        accum_result += exp(input_data.x);
    } else if (remain_channels == 3) {
        accum_result += exp(input_data.x);
    }

    int cur_out_width_pos  = mad24(channel_block_idx, output_width, width_idx);
    input_data = RI_F(input, SAMPLER, (int2)(cur_out_width_pos, batch_height_idx)) - float_max_value;
    const int output_remain = output_channels - mul24(channel_block_idx, 4);

    if (output_remain == 1) {
        input_data.x = exp(input_data.x) / accum_result;
    } else if (output_remain == 2) {
        input_data.y = exp(input_data.y) / accum_result;
        input_data.x = exp(input_data.x) / accum_result;
    } else if (output_remain == 3) {
        input_data.z = exp(input_data.z) / accum_result;
        input_data.y = exp(input_data.y) / accum_result;
        input_data.x = exp(input_data.x) / accum_result;
    } else{
        input_data = exp(input_data) / accum_result;
    }

    WI_F(output, (int2)(cur_out_width_pos, batch_height_idx), input_data);

}
