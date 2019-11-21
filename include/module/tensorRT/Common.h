#ifndef _COMMON_H_
#define _COMMON_H_

#include <NvInfer.h>
#include <iostream>
#include <assert.h>
#include <cudnn.h>
#include <string.h>
namespace bdavs {
// CUDA: use 512 threads per block
#define TRT_CUDA_NUM_THREADS 512

// CUDA: number of blocks for threads.
#define TRT_GET_BLOCKS(N) (N + TRT_CUDA_NUM_THREADS - 1) / TRT_CUDA_NUM_THREADS

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define FLOAT_MAX 3.402823466e+38F        /* max value */

#define clip(x, a, b) x >= a ? (x < b ? x : b-1) : a

#define CUDNNCHECK(status) { if (status != 0) throw std::runtime_error(__FILE__ +  __LINE__ + std::string{"CUDA Error: "} + std::to_string(status)); }

/**
 * Evaluates to true on failure
 * @ingroup util
 */
#define CUDA_FAILED(x)			(CUDA(x) != cudaSuccess)

#define CUDA(x)				cudaCheckError((x), #x, __FILE__, __LINE__)

#define LOG_CUDA "[cuda]   "

/**
 * iDivUp
 * @ingroup util
 */
inline __device__ __host__ int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//
#define CHECK(status)                         \
    do                                            \
{                                             \
    auto ret = (status);                      \
    if (ret != 0)                             \
{                                         \
    std::cout << "Cuda error: " << ret << " - " << cudaGetErrorString(status) << std::endl; \
    abort();                              \
    }                                         \
    } while (0)


inline cudaError_t cudaCheckError(cudaError_t retval, const char* txt, const char* file, int line)
{
#if !defined(CUDA_TRACE)
    if (retval == cudaSuccess)
        return cudaSuccess;
#endif

    printf(LOG_CUDA "%s\n", txt);

    if (retval != cudaSuccess)
    {
        printf(LOG_CUDA "   %s (error %u) (hex 0x%02X)\n", cudaGetErrorString(retval), retval, retval);
        printf(LOG_CUDA "   %s:%i\n", file, line);
    }

    return retval;
}
/**
 *
 * @param cpuPtr
 * @param gpuPtr
 * @param size
 * @return
 */
inline bool cudaAllocMapped(void** cpuPtr, void** gpuPtr, size_t size)
{

    if (!cpuPtr || !gpuPtr || size == 0)
        return false;

    //CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));

    CHECK(cudaHostAlloc(cpuPtr, size, cudaHostAllocMapped));

    CHECK(cudaHostGetDevicePointer(gpuPtr, *cpuPtr, 0));

    memset(*cpuPtr, 0, size);
    printf("[TensorNet] cudaAllocMapped %zu bytes, CPU %p GPU %p\n", size, *cpuPtr, *gpuPtr);

    return true;
}

/**
 * @brief The Logger for TensorRT info/warning/errors
 */
class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kWARNING)
        : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        //if (severity > reportableSeverity)
        //    return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
        case Severity::kERROR: std::cerr << "ERROR: "; break;
        case Severity::kWARNING: std::cerr << "WARNING: "; break;
        case Severity::kINFO: std::cerr << "INFO: "; break;
        default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity;
};
}
#endif //_COMMON_H_
