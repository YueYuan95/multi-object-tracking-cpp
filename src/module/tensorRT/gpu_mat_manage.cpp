#include"gpu_mat_manage.h"
#include<cuda_runtime.h>
#include <iostream>
namespace bdavs {
    GPUDataSingleDevice::GPUDataSingleDevice() {
        debug_new_count = 0;
        debug_free_count = 0;
        for (int i = 0; i < MEM_BIG_GPU_MAT_FIFO_CY; i++) {
            unsigned char *data = NULL;
            cudaMalloc(&(data), MAX_BIG_IMAGE_SIZE * sizeof(unsigned char));
            // std::cout << "************ :" << (void*)data << std::endl;
            m_big_gpu_data_queue.push(data);
            long key=(long)data;
            m_data_number[key]=0;
        }
    }

    int GPUDataSingleDevice::newGPUData(unsigned char *&data, int length) {
        if (length <= MAX_BIG_IMAGE_SIZE) {
            if (m_big_gpu_data_queue.size() > 0) {
                // std::cout << "gpu_mat_manage pop before :" << m_big_gpu_data_queue.size();
                std::lock_guard <std::mutex> lg(m_big_mutex);
                data = m_big_gpu_data_queue.front();
                m_big_gpu_data_queue.pop();
                long key=(long)data;
                m_data_number[key]=1;
//                std::cout<<"new key:"<<key<<std::endl;
                // std::cout << "pop************ :" << (void*)data << std::endl;
                // std::cout << "      after :" << m_big_gpu_data_queue.size() << std::endl;
                debug_new_count++;
            } else {
                data = NULL;
                return -1;
            }
        } else {
            data = NULL;
            return -1;
//            LOG(ERROR) << "gpu data size must small than:" << MAX_BIG_IMAGE_SIZE;
        }
        return 0;
    }

    int GPUDataSingleDevice::freeGPUData(unsigned char *&data, int length) {
        if (length <= MAX_BIG_IMAGE_SIZE) {
            std::lock_guard <std::mutex> lg(m_big_mutex);
             //std::cout << "gpu_mat_manage push before :" << m_big_gpu_data_queue.size()<<std::endl;
            long key=(long)data;
            m_data_number[key]--;
//            std::cout<<"freeGPUData key:"<<key<<std::endl;
//            std::cout<<"m_data_number[key]:"<<m_data_number[key]<<std::endl;
            if (m_data_number[key]==0)
                m_big_gpu_data_queue.push(data);
        //    std::cout <<  "push************ :" << (void*)data << std::endl;
//            std::cout << "      after :" << m_big_gpu_data_queue.size() << std::endl;
//            LOG(WARNING) << "free gpu big_data:" << m_big_gpu_data_queue.size();
            // std::cout<<"keyframe-free gpu big_data:" << m_big_gpu_data_queue.size()<<std::endl;
        } else {
            return -1;
        }

        debug_free_count++;
        return 0;
    }
    int GPUDataSingleDevice::addNumberGPUData(unsigned char * data){
        long key=(long)data;
        m_data_number[key]++;
//        std::cout<<"addNumberGPUData key:"<<key<<std::endl;
    }
}