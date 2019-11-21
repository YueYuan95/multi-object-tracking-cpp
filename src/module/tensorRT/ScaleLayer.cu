
#include "Common.h"
namespace bdavs {
__global__ void ScaleKernel(const int n, const float* in, const float* scale,
                            const int scale_dim, const int inner_dim, float* out)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        const int scale_index = (index / inner_dim) % scale_dim;
        out[index] = in[index] * scale[scale_index];
    }
}

__global__ void ScaleBiasKernel(const int n, const float* in, const float* scale,
                                const float* bias, const int scale_dim, const int inner_dim, float* out)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        const int scale_index = (index / inner_dim) % scale_dim;
        out[index] = in[index] * scale[scale_index] + bias[scale_index];
    }
}

void ScaleForward(const float* bottom_data, const float* scale_data, int count, int scale_dim, int inner_dim, float* top_data)
{
    // NOLINT_NEXT_LINE(whitespace/operators)
    ScaleKernel<< <TRT_GET_BLOCKS(count), TRT_CUDA_NUM_THREADS >> >(count, bottom_data, scale_data, scale_dim, inner_dim, top_data);

    CHECK(cudaDeviceSynchronize());
    }
}
