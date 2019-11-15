
#include <vector>
#include "Common.h"
namespace bdavs {
__global__ void SliceKernel(const int nthreads, const float* in_data,
                            const int num_slices, const int slice_size,
                            const int bottom_slice_axis, const int top_slice_axis,
                            const int offset_slice_axis, float* out_data)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        const int total_slice_size = slice_size * top_slice_axis;
        const int slice_num = index / total_slice_size;
        const int slice_index = index % total_slice_size;
        const int bottom_index = slice_index + (slice_num * bottom_slice_axis + offset_slice_axis) * slice_size;

        out_data[index] = in_data[bottom_index];
    }
}

void SliceForward(const float* bottom_data, int num_slices, int slice_size, int bottom_slice_axis, int top_slice_axis,
                  std::vector<float*>& top)
{

    int offset_slice_axis = 0;

    for (int i = 0; i < top.size(); ++i)
    {
        float* top_data = top[i];
        const int top_slice_size = top_slice_axis * slice_size;
        const int nthreads = top_slice_size * num_slices;
        SliceKernel<<<TRT_GET_BLOCKS(nthreads), TRT_CUDA_NUM_THREADS>>>(nthreads,
                                                                            bottom_data,
                                                                            num_slices,
                                                                            slice_size,
                                                                            bottom_slice_axis,
                                                                            top_slice_axis,
                                                                            offset_slice_axis,
                                                                            top_data);

        CHECK(cudaDeviceSynchronize());

        offset_slice_axis += top_slice_axis;
    }
}
}