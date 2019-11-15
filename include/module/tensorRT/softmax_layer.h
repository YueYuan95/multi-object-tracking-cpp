#ifndef _SOFTMAX_LAYER_H_
#define _SOFTMAX_LAYER_H_

#include <cassert>
#include <NvInferPlugin.h>

//#include "cuda_utility.h"
#include "Common.h"
using namespace nvinfer1;
using namespace plugin;
namespace bdavs {
void cudaSoftmax(int count, int channels, float* bottom, float* top, float* scale_data);

template<int num_classes>
class SoftmaxLayer : public IPlugin {
public:
    SoftmaxLayer() {}

    SoftmaxLayer(const void* buffer, size_t length) {
//        std::cout<<"**SoftmaxLayer"<<std::endl;
        assert(length == sizeof(mCopySize));
        mCopySize = *reinterpret_cast<const size_t*>(buffer);

        CHECK(cudaMalloc((void**)&scale_data, mCopySize / num_classes));
        CHECK(cudaMemset((void*)scale_data, 0, mCopySize / num_classes));
    }

    ~SoftmaxLayer() {
//        std::cout<<"**~SoftmaxLayer"<<std::endl;
        CHECK(cudaFree(scale_data));
    }

    int getNbOutputs() const override {
//        std::cout<<"**getNbOutputs"<<std::endl;
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
//        std::cout<<"**getOutputDimensions"<<std::endl;
        assert(nbInputDims == 1);
        assert(index == 0);
        assert(inputs[index].nbDims == 3);
//        std::cout<<"getOutputDimensions"<<std::endl;
//        std::cout<<inputs[0].d[0]<<std::endl;
//        std::cout<<inputs[0].d[1]<<std::endl;
//        std::cout<<inputs[0].d[2]<<std::endl;
        return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    }

    int initialize() override {
//        std::cout<<"**initialize"<<std::endl;
        return 0;
    }

    void terminate() override {
//        std::cout<<"**terminate"<<std::endl;
    }

    size_t getWorkspaceSize(int maxBatchSize) const override {
        return mCopySize * maxBatchSize;
    }

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override {
        if (scale_data== nullptr)
        {
            CHECK(cudaMalloc((void**)&scale_data, mCopySize / num_classes));
            CHECK(cudaMemset((void*)scale_data, 0, mCopySize / num_classes));
        }
        cudaSoftmax(batchSize* mCopySize / sizeof(float), num_classes, (float*)(*inputs), static_cast<float*>(*outputs), scale_data);
        return 0;
    }

    void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSzie) override {
        mCopySize = inputDims[0].d[0] * inputDims[0].d[1] * inputDims[0].d[2] * sizeof(float);
    }

    size_t getSerializationSize() override {
        return sizeof(mCopySize);
    }

    void serialize(void *buffer) override {
        *reinterpret_cast<size_t*>(buffer) = mCopySize;
    }

protected:
    size_t mCopySize;
    float* scale_data = nullptr;
};
}
#endif //_SOFTMAX_LAYER_H_
