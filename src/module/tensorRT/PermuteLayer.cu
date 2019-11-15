
#include "Common.h"
namespace bdavs {
__global__ void PermuteKernel(const int nthreads,
                              const float* bottom_data, const int* permute_order,
                              const int* old_steps, const int* new_steps, const int num_axes,
                              float* top_data)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        int temp_idx = index;
        int old_idx = 0;
        //printf("input: %f, index: %d  order %d\n", bottom_data[index], index);
        for (int i = 0; i < num_axes; ++i)
        {
            int order = permute_order[i];
            old_idx += (temp_idx / new_steps[i]) * old_steps[order];
            temp_idx %= new_steps[i];
            //printf("order: %d  old_idx: %d  temp_idx: %d  num_axes: %d\n",
            //       order, old_idx, temp_idx, i);
        }

        top_data[index] = bottom_data[old_idx];
    }
}

void PermuteForward(const float* bottom_data, int count, int* permute_order, int* new_steps, int* old_steps, int num_axes_, int need_permute_,
                    float* top_data)
{
    if (need_permute_)
    {
        // NOLINT_NEXT_LINE(whitespace/operators)
        PermuteKernel<<<TRT_GET_BLOCKS(count), TRT_CUDA_NUM_THREADS>>>(count, bottom_data, permute_order, old_steps, new_steps,
                                                                           num_axes_, top_data);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaPeekAtLastError());
    }
    else
    {
        // If there is no need to permute, we share data to save memory.
        CHECK(cudaMemcpy(top_data, bottom_data, count*sizeof(float), cudaMemcpyDeviceToDevice));
        }
    }
}
