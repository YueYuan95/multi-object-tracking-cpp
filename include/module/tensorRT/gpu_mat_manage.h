#ifndef _GPU_MAT_MANAGE_H_
#define _GPU_MAT_MANAGE_H_
#include <vector>
#include <mutex>
#include <queue>
#include<map>
#include <opencv2/core/core.hpp>
#include <atomic>
#include <sys/time.h>
#define PIXMAP_SIZE 4
#define MAX_BIG_IMAGE_SIZE (1920*1080*PIXMAP_SIZE)

#define MEM_BIG_GPU_MAT_FIFO_CY    (350)


namespace bdavs {
    class GPUDataSingleDevice {
    private:
        int debug_new_count;
        int debug_free_count;
    public:
        std::queue<unsigned char *> m_big_gpu_data_queue;
        std::mutex m_big_mutex;
        std::mutex m_medium_mutex;
        std::mutex m_small_mutex;
        std::map<long,int> m_data_number;
        // std::atomic<int> m_gpu_data_create_num;
        GPUDataSingleDevice();

        int newGPUData(unsigned char *&data, int length);

        int freeGPUData(unsigned char *&data, int length);
        int addNumberGPUData(unsigned char *data);
    };
}
#endif