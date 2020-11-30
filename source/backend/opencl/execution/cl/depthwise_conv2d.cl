
__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void depthwise_conv2d_s1(GLOBAL_SIZE_2_DIMS(0) __read_only image2d_t input, __read_only image2d_t filter,
                                  __read_only image2d_t bias,
                                  __write_only image2d_t output,
                                  __private const int2 input_shape,
                                  __private const int inChannelBlocks, 
                                  __private const int2 outputShape,
                                  __private const int2 filterShape,
                                  __private const int2 paddingShape) {

    const int outChannelWidthIdx = get_global_id(0);
    const int outHeightBlockIdx     = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(0, outChannelWidthIdx, outHeightBlockIdx);
    int ow4              = (outputShape.y + 3) / 4;
    const int outChannelBlockIdx = outChannelWidthIdx / ow4;
    const int outWidthBlockidx   = outChannelWidthIdx % ow4;

    const int inChannelBlockIdx = outChannelBlockIdx;

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(outChannelBlockIdx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    const int outWidthBlockidx4 = outWidthBlockidx << 2;
    const int in_width0             = outWidthBlockidx4 - paddingShape.y;
    const int in_width1             = in_width0 + 1;
    const int in_width2             = in_width0 + 2;
    const int in_width3             = in_width0 + 3;

    int heightIdx            = outHeightBlockIdx % outputShape.x - paddingShape.x;
    const int outBatchIdx = mul24((outHeightBlockIdx / outputShape.x), input_shape.x);
    const int in_idx = mul24(inChannelBlockIdx, input_shape.y);

    const int inWidthIdx0 = select(in_idx + in_width0, -1, (in_width0 < 0 || in_width0 >= input_shape.y));
    const int inWidthIdx1 = select(in_idx + in_width1, -1, (in_width1 < 0 || in_width1 >= input_shape.y));
    const int inWidthIdx2 = select(in_idx + in_width2, -1, (in_width2 < 0 || in_width2 >= input_shape.y));

    FLOAT4 in0, in1, in2, in3;
    for (int kh = 0; kh < filterShape.x; kh++) {
        int in_hb_value = select(heightIdx + outBatchIdx, -1, (heightIdx < 0 || heightIdx >= input_shape.x));
        heightIdx++;
        in1       = RI_F(input, SAMPLER, (int2)(inWidthIdx0, in_hb_value));
        in2       = RI_F(input, SAMPLER, (int2)(inWidthIdx1, in_hb_value));
        in3       = RI_F(input, SAMPLER, (int2)(inWidthIdx2, in_hb_value));
        for (int kw = 0; kw < filterShape.y; kw++) {

            int filterIdx   = mad24(kh, filterShape.y, kw);
            in0 = in1;
            in1 = in2;
            in2 = in3;

            int inWidthIdx = in_width3 + kw;
            inWidthIdx = select(in_idx + inWidthIdx, -1, (inWidthIdx < 0 || inWidthIdx >= input_shape.y));
            in3  = RI_F(input, SAMPLER, (int2)(inWidthIdx, in_hb_value));

            FLOAT4 weights = RI_F(filter, SAMPLER, (int2)(filterIdx, inChannelBlockIdx));

            out0 = mad(in0, weights, out0);
            out1 = mad(in1, weights, out1);
            out2 = mad(in2, weights, out2);
            out3 = mad(in3, weights, out3);
        }
    }

#ifdef RELU
    out0 = fmax(out0, (FLOAT4)0);
    out1 = fmax(out1, (FLOAT4)0);
    out2 = fmax(out2, (FLOAT4)0);
    out3 = fmax(out3, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
    out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);
    out2 = clamp(out2, (FLOAT4)0, (FLOAT4)6);
    out3 = clamp(out3, (FLOAT4)0, (FLOAT4)6);
#endif

    const int remain     = outputShape.y - outWidthBlockidx4;
    int outWidthIdx       = mul24(outChannelBlockIdx, outputShape.y) + outWidthBlockidx4;
    if (remain >= 4) {
        WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), out0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), out1);
        WI_F(output, (int2)(outWidthIdx + 2, outHeightBlockIdx), out2);
        WI_F(output, (int2)(outWidthIdx + 3, outHeightBlockIdx), out3);
    } else if (remain == 3) {
        WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), out0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), out1);
        WI_F(output, (int2)(outWidthIdx + 2, outHeightBlockIdx), out2);
    } else if (remain == 2) {
        WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), out0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), out1);
    } else if (remain == 1) {
        WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), out0);
    }
}

__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void depthwise_conv2d(GLOBAL_SIZE_2_DIMS(0) __read_only image2d_t input, __read_only image2d_t filter,
                               __read_only image2d_t bias,
                               __write_only image2d_t output,
                               __private const int2 input_shape,
                               __private const int inChannelBlocks, __private const int2 outputShape,
                               __private const int2 filterShape,
                               __private const int2 paddingShape,
                               __private const int2 dilationShape,
                               __private const int2 strideShape) {

    const int outChannelWidthIdx = get_global_id(0);
    const int outHeightIdx     = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(0, outChannelWidthIdx, outHeightIdx);

    int ow4              = (outputShape.y + 3) / 4;
    const int outChannelBlockIdx = outChannelWidthIdx / ow4;
    const int outWidthBlockidx   = outChannelWidthIdx % ow4;

    const int inChannelBlockIdx = outChannelBlockIdx;

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(outChannelBlockIdx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    const int in_width0  = mad24(outWidthBlockidx, strideShape.y << 2, -paddingShape.y);
    const int in_width1  = in_width0 + strideShape.y;
    const int in_width2  = in_width1 + strideShape.y;
    const int in_width3  = in_width2 + strideShape.y;
    int heightIdx = mad24(outHeightIdx % outputShape.x, strideShape.x, -paddingShape.x);

    const int outBatchIdx = mul24((outHeightIdx / outputShape.x), input_shape.x);

    const int in_idx = mul24(inChannelBlockIdx, input_shape.y);
    for (int kh = 0; kh < filterShape.x; kh++) {
        int in_hb_value = select(heightIdx + outBatchIdx, -1, (heightIdx < 0 || heightIdx >= input_shape.x));
        heightIdx += dilationShape.x;
        for (int kw = 0; kw < filterShape.y; kw++) {
            int filterIdx = mad24(kh, filterShape.y, kw);
            FLOAT4 in0, in1, in2, in3;
            int inWidthIdx = mul24(kw, dilationShape.y);

            READ_INPUT_IMAGE(0, inWidthIdx);
            READ_INPUT_IMAGE(1, inWidthIdx);
            READ_INPUT_IMAGE(2, inWidthIdx);
            READ_INPUT_IMAGE(3, inWidthIdx);

            FLOAT4 weights = RI_F(filter, SAMPLER, (int2)(filterIdx, inChannelBlockIdx));

            out0 = mad(in0, weights, out0);
            out1 = mad(in1, weights, out1);
            out2 = mad(in2, weights, out2);
            out3 = mad(in3, weights, out3);
        }
    }

#ifdef RELU
    out0 = fmax(out0, (FLOAT4)0);
    out1 = fmax(out1, (FLOAT4)0);
    out2 = fmax(out2, (FLOAT4)0);
    out3 = fmax(out3, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
    out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);
    out2 = clamp(out2, (FLOAT4)0, (FLOAT4)6);
    out3 = clamp(out3, (FLOAT4)0, (FLOAT4)6);
#endif

    const int outWidthBlockidx4        = outWidthBlockidx << 2;
    const int remain = outputShape.y - outWidthBlockidx4;
    int outWidthIdx   = mul24(outChannelBlockIdx, outputShape.y) + outWidthBlockidx4;
    if (remain >= 4) {
        WI_F(output, (int2)(outWidthIdx, outHeightIdx), out0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightIdx), out1);
        WI_F(output, (int2)(outWidthIdx + 2, outHeightIdx), out2);
        WI_F(output, (int2)(outWidthIdx + 3, outHeightIdx), out3);
    } else if (remain == 3) {
        WI_F(output, (int2)(outWidthIdx, outHeightIdx), out0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightIdx), out1);
        WI_F(output, (int2)(outWidthIdx + 2, outHeightIdx), out2);
    } else if (remain == 2) {
        WI_F(output, (int2)(outWidthIdx, outHeightIdx), out0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightIdx), out1);
    } else if (remain == 1) {
        WI_F(output, (int2)(outWidthIdx, outHeightIdx), out0);
    }
}
