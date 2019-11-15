#ifndef _FLATTEN_LAYER_H_
#define _FLATTEN_LAYER_H_

#include <cassert>
#include <NvInferPlugin.h>

//#include "cuda_utility.h"
#include "Common.h"
using namespace nvinfer1;
using namespace plugin;
namespace bdavs {
class FlattenLayer : public IPlugin {
public:
    FlattenLayer() {}

    FlattenLayer(const void* buffer, size_t length) {
        assert(length == 3 * sizeof(int));

        const int* d = reinterpret_cast<const int*>(buffer);

        _size = d[0] * d[1] * d[2];

        dimBottom = DimsCHW{ d[0], d[1], d[2] };
    }

    int getNbOutputs() const override {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
        assert(1 == nbInputDims);
        assert(0 == index);
        assert(3 == inputs[index].nbDims);

        _size = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];

        return DimsCHW(_size, 1, 1);
    }

    int initialize() override {
        return 0;
    }

    void terminate() override {

    }

    size_t getWorkspaceSize(int maxBatchSize) const override {
        return 0;
    }

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override {
        CHECK(cudaMemcpyAsync(outputs[0], inputs[0], batchSize * _size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        return 0;
    }

    void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSzie) override {
        dimBottom = DimsCHW(inputDims[0].d[0], inputDims[0].d[1], inputDims[0].d[2]);
    }

    size_t getSerializationSize() override {
        return 3 * sizeof(int);
    }

    void serialize(void *buffer) override {
        int* d = reinterpret_cast<int*>(buffer);

        d[0] = dimBottom.c();
        d[1] = dimBottom.h();
        d[2] = dimBottom.w();
    }

private:
    DimsCHW dimBottom;

    int _size;
};
}
#endif //_SOFTMAX_LAYER_H_
