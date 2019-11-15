#ifndef _LSTM_LAYER_H_
#define _LSTM_LAYER_H_

#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <vector>
#include "Common.h"

using namespace nvinfer1;
using namespace plugin;
namespace bdavs {
void SliceForward(const float* bottom_data, int num_slices, int slice_size, int bottom_slice_axis, int top_slice_axis,
                  std::vector<float*>& top);

void ScaleForward(const float* bottom_data, const float* scale_data, int count, int scale_dim, int inner_dim, float* top_data);

void LSTMUnitForward(const int count, const float* C_prev, const float* X, const float* cont,
                     float* X_acts, float* C, float* H, const int X_count, const int hidden_dim);

void ConcatForward(std::vector<float*>& bottom, const int bottom_concat_axis, const int top_concat_axis,
                   const int concat_input_size, const int num_concats, float* top_data);

class LSTMLayer : public IPlugin
{
public:
    LSTMLayer(const Weights* weights, int nbWeights, int num_output, int time_step);

    LSTMLayer(const void* buffer, size_t length);

    ~LSTMLayer();

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
    Weights copyToDevice(const void* hostData, int count);

    int copyFromDevice(char* hostBuffer, Weights deviceWeights);

private:
    int mInputChannels;
    int mInputHeight;
    int mInputWidth;

    int mOutputChannels;
    int mOutputHeight;
    int mOutputWidth;

    int nbOutputChannels;
    int nbTimeStep;

    Weights mKernelWeights0, mBiasWeights0;
    Weights mKernelWeights1;

    cublasHandle_t mCublas;
};
}
#endif // _PERMUTE_LAYER_H_
