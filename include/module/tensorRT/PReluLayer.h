//
// Created by svt on 19-5-8.
//

#ifndef _PRELU_LAYER_H_
#define _PRELU_LAYER_H_

#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include "Common.h"

using namespace nvinfer1;
using namespace plugin;
namespace bdavs {
void PReLUForward(int count, int channels, int dim, float* bottom_data, float* top_data);

class PReluLayer : public IPlugin
{
public:
    PReluLayer();

    PReluLayer(const void* buffer, size_t length);

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSzie) override;

    size_t getSerializationSize() override;

    void serialize(void *buffer) override;

private:
    int mCount;
    int mChannels;
    int mDim;
};
}
#endif //_PRELU_LAYER_H_
