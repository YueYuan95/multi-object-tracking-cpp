
#ifndef _INNER_PRODUCT_LAYER_H_
#define _INNER_PRODUCT_LAYER_H_

#include <NvInferPlugin.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include "Common.h"

using namespace nvinfer1;
using namespace plugin;
namespace bdavs {
class InnerProductLayer : public IPlugin
{
public:
    InnerProductLayer(const Weights* weights, int nbWeights, int nbOutputChannels, int num_axis);

    InnerProductLayer(const void* buffer, size_t length);

    ~InnerProductLayer();

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
    size_t type2size(DataType type);

    Weights copyToDevice(const void* hostData, int count);

    int copyFromDevice(char* hostBuffer, Weights deviceWeights);

private:
    int mNbNumAxis;

    bool bias_term;

    int mNbOutputChannels, mNbInputChannels;
    int mNbInputHeight, mNbInputWidth;
    Weights mKernelWeights, mBiasWeights;

    cublasHandle_t mCublas;
};
}
#endif //_INNER_PRODUCT_LAYER_H_
