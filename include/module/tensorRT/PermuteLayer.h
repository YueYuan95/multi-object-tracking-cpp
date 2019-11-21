#ifndef _PERMUTE_LAYER_H_
#define _PERMUTE_LAYER_H_

#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include"Common.h"

using namespace nvinfer1;
using namespace plugin;
namespace bdavs {
void PermuteForward(const float* bottom_data, int count, int* permute_order, int* new_steps, int* old_steps, int num_axes_, int need_permute_,
                    float* top_data);

class PermuteLayer : public IPlugin
{
public:
    PermuteLayer(int order0, int order1, int order2);

    PermuteLayer(int order0, int order1, int order2, int order3);

    ~PermuteLayer();

    PermuteLayer(const void* buffer, size_t length);

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
    int mInputChannels;
    int mInputHeight;
    int mInputWidth;

    int mOutputChannels;
    int mOutputHeight;
    int mOutputWidth;

    int num_axes;
    int order[4];

    int* permute_order;
    int* new_steps;
    int* old_steps;
};
}
#endif // _PERMUTE_LAYER_H_
