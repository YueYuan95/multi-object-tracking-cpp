#ifndef _RESHAPE_LAYER_HPP_
#define _RESHAPE_LAYER_HPP_

#include <cassert>
#include <NvInferPlugin.h>

//#include "cuda_utility.h"
#include "Common.h"
using namespace nvinfer1;
using namespace plugin;
namespace bdavs {
template<int num_classes>
class ReshapeLayer : public IPlugin {
public:
    ReshapeLayer() {}

    ReshapeLayer(const void* buffer, size_t length) {
        assert(length == sizeof(mCopySize));
        mCopySize = *reinterpret_cast<const size_t*>(buffer);
    }

    int getNbOutputs() const override {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
        assert(index == 0 && nbInputDims == 1 && inputs[index].nbDims == 3);
        assert((inputs[0].d[0]) * (inputs[0].d[1]) % num_classes == 0);
        return DimsCHW(num_classes, inputs[0].d[0] * inputs[0].d[1] / num_classes, inputs[0].d[2]);
    }

    int initialize() override {
        return 0;
    }

    void terminate() override {

    }

    size_t getWorkspaceSize(int maxBatchSize) const override{
        return 0;
    }

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override {
        CHECK(cudaMemcpyAsync(outputs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice, stream));
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
};
}
#endif //_SOFTMAX_LAYER_H_
