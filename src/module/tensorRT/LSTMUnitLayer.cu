
#include "Common.h"
namespace bdavs {
__device__ float cudaSigmoid(const float x)
{
  return float(1) / (float(1) + exp(-x));
}

__device__ float cudaTanh(const float x)
{
    return float(2) * cudaSigmoid(float(2) * x) - float(1);
}

__global__ void LSTMActsKernel(const int nthreads, const int dim, const float* X, float* X_acts)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        const int x_dim = 4 * dim;
        const int d = index % x_dim;
        if (d < 3 * dim)
        {
            X_acts[index] = cudaSigmoid(X[index]);
        }
        else
        {
            X_acts[index] = cudaTanh(X[index]);
        }
    }
}

__global__ void LSTMUnitKernel(const int nthreads, const int dim,
                               const float* C_prev, const float* X, const float* cont,
                               float* C, float* H)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        const int n = index / dim;
        const int d = index % dim;
        const float* X_offset = X + 4 * dim * n;
        const float i = X_offset[d];
        const float f = X_offset[1 * dim + d];
        const float o = X_offset[2 * dim + d];
        const float g = X_offset[3 * dim + d];
        const float c_prev = C_prev[index];
        const float c = cont[n] * f * c_prev + i * g;
        C[index] = c;
        const float tanh_c = tanh(c);
        H[index] = o * tanh_c;
    }
}

void LSTMUnitForward(const int count, const float* C_prev, const float* X, const float* cont,
                     float* X_acts, float* C, float* H, const int X_count, const int hidden_dim)
{
    LSTMActsKernel<<<TRT_GET_BLOCKS(X_count), TRT_CUDA_NUM_THREADS>>>(X_count, hidden_dim, X, X_acts);

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaPeekAtLastError());

    // NOLINT_NEXT_LINE(whitespace/operators)
    LSTMUnitKernel<<<TRT_GET_BLOCKS(count), TRT_CUDA_NUM_THREADS>>>(count, hidden_dim, C_prev, X_acts, cont, C, H);

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaPeekAtLastError());
    }
}
