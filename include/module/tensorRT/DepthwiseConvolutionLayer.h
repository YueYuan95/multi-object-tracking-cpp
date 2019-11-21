#ifndef __DEPTHWISE_CONVOLUTION_LAYER_H__
#define __DEPTHWISE_CONVOLUTION_LAYER_H__

#include <memory>

// Cuda
#include <cuda_runtime_api.h>

// TensorRT
#include <NvInferPlugin.h>

#include"Common.h"


using namespace nvinfer1;
namespace bdavs {
void DepthwiseConvolutionForward(int count, int num, int channels, int top_height, int top_width, int bottom_height, int bottom_width,
                                 int kernel_h_, int kernel_w_, int stride_h_, int stride_w_, int pad_h_, int pad_w_, int dilation_h_, int dilation_w_,
                                 const float* bottom_data, float* top_data, const float* weight_data, const float* bias_data, const bool bias_term);

typedef struct
{
    int kernel_h;
    int kernel_w;
    int pad_h;
    int pad_w;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;

} DWConvolutionParam;

class DepthwiseConvolutionLayer: public IPlugin
{
public:
    DepthwiseConvolutionLayer(const Weights* weights, int nbWeights, DWConvolutionParam& params);

    DepthwiseConvolutionLayer(const void* buffer, size_t length);

    ~DepthwiseConvolutionLayer();

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override;

    int enqueue(int batchSize, const void * const *inputs, void **outputs, void*, cudaStream_t stream) override;

    void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSzie) override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int initialize() override;
    void terminate() override;

    size_t getSerializationSize() override;
    void serialize(void *buffer) override;

private:
    Weights copyToDevice(const void* hostData, int count);

    int copyFromDevice(char* hostBuffer, Weights deviceWeights);

private:
    Weights mWeights;
    Weights mBias;

    int input_channels;
    int input_width;
    int input_height;

    size_t output_count;
    int output_width;
    int output_height;

    bool bias_term;

    int kernel_h_;
    int kernel_w_;
    int pad_h_;
    int pad_w_;
    int stride_h_;
    int stride_w_;
    int dilation_h_;
    int dilation_w_;
};
}
#endif //__DEPTHWISE_CONVOLUTION_LAYER_H__

