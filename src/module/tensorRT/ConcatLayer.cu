
#include <vector>
#include "Common.h"
namespace bdavs {
__global__ void ConcatKernel(const int nthreads, const float* in_data, const int concat_size,
                             const int top_concat_axis, const int bottom_concat_axis,
                             const int offset_concat_axis, float* out_data)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        const int total_concat_size = concat_size * bottom_concat_axis;
        const int concat_num = index / total_concat_size;
        const int concat_index = index % total_concat_size;
        const int top_index = concat_index + (concat_num * top_concat_axis + offset_concat_axis) * concat_size;

        out_data[top_index] = in_data[index];
    }
}

void ConcatForward(std::vector<float*>& bottom, const int bottom_concat_axis, const int top_concat_axis,
                   const int concat_input_size, const int num_concats, float* top_data)
{
    int offset_concat_axis = 0;
    for (int i = 0; i < bottom.size(); ++i)
    {
        const int bottom_concat_size = bottom_concat_axis * concat_input_size;
        const int nthreads = bottom_concat_size * num_concats;
        ConcatKernel<<<TRT_GET_BLOCKS(nthreads), TRT_CUDA_NUM_THREADS>>>(nthreads, bottom[i], concat_input_size,
                                                                             top_concat_axis, bottom_concat_axis, offset_concat_axis, top_data);
        offset_concat_axis += bottom_concat_axis;
    }
}
}
