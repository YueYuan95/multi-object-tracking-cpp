
#include "Common.h"
namespace bdavs {
__global__ void PReLUKernel(int count, int channels, int dim, float* bottom_data, float* top_data, float* slope_data, int div_factor)
{
    CUDA_KERNEL_LOOP(index, count)
    {
      int c = (index / dim) % channels / div_factor;
      top_data[index] = bottom_data[index] > 0 ? bottom_data[index] : bottom_data[index] * slope_data[c];
    }
}

void PReLUForward(int count, int channels, int dim, float* bottom_data, float* top_data)
{
    bool channel_shared_ = false;

    float* slope_data = nullptr;
    CHECK(cudaMalloc<float>(&slope_data, channels*sizeof(float)));

    const int div_factor = channel_shared_ ? channels : 1;

    // NOLINT_NEXT_LINE(whitespace/operators)
    PReLUKernel<<<TRT_GET_BLOCKS(count), TRT_CUDA_NUM_THREADS>>>(
               count, channels, dim, bottom_data, top_data, slope_data, div_factor);

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(slope_data));

    CHECK(cudaPeekAtLastError());
}
}