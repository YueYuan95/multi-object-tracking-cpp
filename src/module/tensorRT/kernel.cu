/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "Common.h"

#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include <vector>
namespace bdavs {
// CUDA: use 512 threads per block
#define CAFFE_CUDA_NUM_THREADS 512

// CUDA: number of blocks for threads.
#define CAFFE_GET_BLOCKS(N) (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define FLT_MAX 3.402823466e+38F        /* max value */

__global__ void kernel_channel_max(const int num, const int channels, const int spatial_dim, const float* data, float* out)
{
    CUDA_KERNEL_LOOP(index, num * spatial_dim)
    {
        int n = index / spatial_dim;
        int s = index % spatial_dim;
        float maxval = -FLT_MAX;
        for (int c = 0; c < channels; ++c)
        {
            maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
        }
        out[index] = maxval;
    }
}


__global__ void kernel_channel_subtract(const int count, const int num, const int channels, const int spatial_dim, const float* channel_max, float* data)
{
    CUDA_KERNEL_LOOP(index, count)
    {
        int n = index / channels / spatial_dim;
        int s = index % spatial_dim;
        data[index] -= channel_max[n * spatial_dim + s];
    }
}


__global__ void kernel_exp(const int count, const float* data, float* out)
{
    CUDA_KERNEL_LOOP(index, count)
    {
        out[index] = exp(data[index]);
    }
}


__global__ void kernel_channel_sum(const int num, const int channels, const int spatial_dim, const float* data, float* channel_sum)
{
    CUDA_KERNEL_LOOP(index, num * spatial_dim)
    {
        int n = index / spatial_dim;
        int s = index % spatial_dim;
        float sum = 0;
        for (int c = 0; c < channels; ++c)
        {
            sum += data[(n * channels + c) * spatial_dim + s];
        }
        channel_sum[index] = sum;
    }
}

__global__ void kernel_channel_div(const int count, const int num, const int channels, const int spatial_dim, const float* channel_sum, float* data)
{
    CUDA_KERNEL_LOOP(index, count)
    {
        int n = index / channels / spatial_dim;
        int s = index % spatial_dim;
        data[index] /= channel_sum[n * spatial_dim + s];
    }
}


void cudaSoftmax(int count, int channels, float* bottom, float* top, float* scale_data)
{
    int outer_num = count / channels;
    CHECK(cudaMemcpy(top, bottom, sizeof(float)*count, cudaMemcpyDefault));

    // We need to subtract the max to avoid numerical issues, compute the exp,
    // and then normalize.
    // compute max
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_max<<<CAFFE_GET_BLOCKS(outer_num), CAFFE_CUDA_NUM_THREADS>>>(outer_num, channels, 1, top, scale_data);
    cudaDeviceSynchronize();

    // subtract
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_subtract<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, outer_num, channels, 1, scale_data, top);
    cudaDeviceSynchronize();

    // exponentiate
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_exp<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top, top);
    cudaDeviceSynchronize();

    // sum after exp
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_sum<<<CAFFE_GET_BLOCKS(outer_num), CAFFE_CUDA_NUM_THREADS>>>(outer_num, channels, 1, top, scale_data);
    cudaDeviceSynchronize();

    // divide
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_div<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, outer_num, channels, 1, scale_data, top);
    cudaDeviceSynchronize();
}


/**
 * The depthwise layer for mobilenet.   only for stride 1
 */
__global__ void ConvForward(const int nthreads,
                            const float* bottom_data, const int channels, const int height, const int width,
                            const int conved_height, const int conved_width,
                            const int kernel_h, const int kernel_w,
                            const int stride_h, const int stride_w,
                            const int pad_h, const int pad_w,
                            float* top_data,
                            const float* weight,
                            const float* bias
                            )
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        const int pw = index % conved_width;
        const int ph = (index / conved_width) % conved_height;
        const int c = (index / conved_width / conved_height) % channels;
        const int n = index / conved_width / conved_height / channels;

        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;

        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);

        hstart = max(hstart, 0);
        wstart = max(wstart, 0);

        hend = min(hend, height);
        wend = min(wend, width);

        float aveval = 0;

        const float* const bottom_slice = bottom_data + (n * channels + c) * height * width;
        const float* const weight_slice = weight + c * kernel_h * kernel_w;

        int khstart = hend < kernel_h ? kernel_h - hend : 0;
        int kwstart = wend < kernel_w ? kernel_w - wend : 0;
        for (int h = hstart; h < hend; ++h)
        {
            for (int w = wstart; w < wend; ++w)
            {
                aveval += bottom_slice[h * width + w] * weight_slice[(khstart + h - hstart) * kernel_w + (kwstart + w - wstart)];
            }
        }

        aveval += bias[c];

        top_data[index] = aveval;
    }
}

void depthwiseConvolutionForward(size_t count,
                                 const float* input, int channels, int intputHeight, int inputWidth,
                                 int kSize,
                                 int stride,
                                 int pad,
                                 float* output, int outputHeight, int outputWidth,
                                 const float* weights,
                                 const float* bias
                                 )
{
    //CHECK(cudaMalloc((void**)&output, count*sizeof(float)));
    //CHECK(cudaMemset((void*)output, 0, count*sizeof(float)));

    ConvForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
                                                                     input, channels, intputHeight, inputWidth,
                                                                     outputHeight, outputWidth,
                                                                     kSize, kSize,
                                                                     stride, stride,
                                                                     pad, pad,
                                                                     output,
                                                                     weights,
                                                                     bias
                                                                     );
    cudaDeviceSynchronize();
}

// Bi-linear interpolation
// IN : [channels height1 width1] cropped from a bigger [Height1 Width1] image
// OUT: [channels height2 width2] cropped from a bigger [Height2 Width2] image
__global__ void caffe_gpu_interp2_kernel(const int n, const float rheight, const float rwidth,
                                         const int channels,
                                         const float *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
                                         float *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
    {
        const int channel_all=index / (width2*height2);
        const int index_channel_frame= index%(width2*height2);
        const int w2 = index_channel_frame % width2; // 0:width2-1
        const int h2 = index_channel_frame / width2; // 0:height2-1
        const float h1r = rheight * h2;
        const int h1 = h1r;
        const int h1p = (h1 < height1 - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = float(1.) - h1lambda;
        //
        const float w1r = rwidth * w2;
        const int w1 = w1r;
        const int w1p = (w1 < width1 - 1) ? 1 : 0;
        const float w1lambda = w1r - w1;
        const float w0lambda = float(1.) - w1lambda;
        //
        const float* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
        float* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
        //for (int c = 0; c < channels; ++c)
        const int c=channel_all;
        {
            pos1 += Width1 * Height1*c;
            pos2 += Width2 * Height2*c;
            pos2[0] =
                    h0lambda * (w0lambda * pos1[0]            + w1lambda * pos1[w1p]) +
                    h1lambda * (w0lambda * pos1[h1p * Width1] + w1lambda * pos1[h1p * Width1 + w1p]);
            
        }
    }
}

void caffe_gpu_interp2(const int channels,
                       const float *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
                       float *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2)
{
    assert(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0 && x2 >= 0 && y2 >= 0 && height2 > 0 && width2 > 0);
    assert(Width1 >= width1 + x1 && Height1 >= height1 + y1 && Width2 >= width2 + x2 && Height2 >= height2 + y2);
    const float rheight = (height2 > 1) ? static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
    const float rwidth = (width2 > 1) ? static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
    const int num_kernels = height2 * width2*channels;
    caffe_gpu_interp2_kernel<<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(num_kernels, rheight, rwidth, channels,
                                                                                        data1, x1, y1, height1, width1, Height1, Width1,
                                                                                        data2, x2, y2, height2, width2, Height2, Width2);
    CHECK(cudaPeekAtLastError());
    }
}
