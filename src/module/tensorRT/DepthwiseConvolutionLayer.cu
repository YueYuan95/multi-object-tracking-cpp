
#include "Common.h"

namespace bdavs {
__global__ void ConvolutionDepthwiseWeightKernel(const int nthreads,
                                                 const float* const bottom_data, const float* const weight_data,
                                                 const int num, const int channels, const int top_height,
                                                 const int top_width, const int bottom_height, const int bottom_width,
                                                 const int kernel_h, const int kernel_w, const int stride_h,
                                                 const int stride_w, const int pad_h, const int pad_w,
                                                 const int dilation_h, const int dilation_w, float* const top_data)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        const int n = index / channels / top_height / top_width;
        const int c = (index / top_height / top_width) % channels;
        const int h = (index / top_width) % top_height;
        const int w = index % top_width;
        const float* weight = weight_data + c * kernel_h * kernel_w;
        float value = 0;
        for (int kh = 0; kh < kernel_h; ++kh)
        {
            for (int kw = 0; kw < kernel_w; ++kw)
            {
                const int h_in = -pad_h + h * stride_h + kh * dilation_h;
                const int w_in = -pad_w + w * stride_w + kw * dilation_w;
                if ((h_in >= 0) && (h_in < bottom_height) && (w_in >= 0) && (w_in < bottom_width))
                {
                    const int offset = ((n * channels + c) * bottom_height + h_in) * bottom_width + w_in;
                    value += (*weight) * bottom_data[offset];
                }
                ++weight;
            }
        }
        top_data[index] = value;
    }
}


__global__ void ConvolutionDepthwiseBiasKernel(const int nthreads,
                                               const float* const bias_data, const int num, const int channels,
                                               const int top_height, const int top_width, float* const top_data)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        const int c = (index / top_height / top_width) % channels;
        top_data[index] += bias_data[c];
    }
}


void DepthwiseConvolutionForward(int count, int num, int channels, int top_height, int top_width, int bottom_height, int bottom_width,
                                 int kernel_h_, int kernel_w_, int stride_h_, int stride_w_, int pad_h_, int pad_w_, int dilation_h_, int dilation_w_,
                                 const float* bottom_data, float* top_data, const float* weight_data, const float* bias_data, const bool bias_term)
{

    ConvolutionDepthwiseWeightKernel<<<TRT_GET_BLOCKS(count), TRT_CUDA_NUM_THREADS>>>(count, bottom_data, weight_data, num, channels,
                                                                                          top_height, top_width, bottom_height, bottom_width,
                                                                                          kernel_h_, kernel_w_, stride_h_, stride_w_,
                                                                                          pad_h_, pad_w_, dilation_h_, dilation_w_, top_data);

    CHECK(cudaDeviceSynchronize());

    if (bias_term)
    {
        ConvolutionDepthwiseBiasKernel<<<TRT_GET_BLOCKS(count), TRT_CUDA_NUM_THREADS>>>(count, bias_data, num, channels,
                                                                                            top_height, top_width, top_data);
        CHECK(cudaDeviceSynchronize());
        }
    }
}
