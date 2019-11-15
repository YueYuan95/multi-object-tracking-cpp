
#include "DepthwiseConvolutionLayer.h"
namespace bdavs {
DepthwiseConvolutionLayer::DepthwiseConvolutionLayer(const Weights* weights, int nbWeights, DWConvolutionParam& params)
{
    bias_term = nbWeights == 2 ? true : false;

    if(bias_term)
    {
        mWeights = copyToDevice(weights[0].values, weights[0].count);
        mBias = copyToDevice(weights[1].values, weights[1].count);
    }
    else
    {
        mWeights = copyToDevice(weights[0].values, weights[0].count);
    }

    kernel_h_ = params.kernel_h;
    kernel_w_ = params.kernel_w;
    pad_h_ = params.pad_h;
    pad_w_ = params.pad_w;
    stride_h_ = params.stride_h;
    stride_w_ = params.stride_w;
    dilation_h_ = params.dilation_h;
    dilation_w_ = params.dilation_w;
}

DepthwiseConvolutionLayer::DepthwiseConvolutionLayer(const void* buffer, size_t length)
{
    const char* d = reinterpret_cast<const char*>(buffer);
    const char* a = d;
    input_channels = *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    input_height = *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    input_width = *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    output_count = *reinterpret_cast<const size_t*>(d);
    d += sizeof(size_t);
    output_height =  *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    output_width = *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    kernel_h_ = *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    kernel_w_ = *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    pad_h_ = *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    pad_w_ = *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    stride_h_ = *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    stride_w_ = *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    dilation_h_ = *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    dilation_w_ = *reinterpret_cast<const int*>(d);
    d += sizeof(int);

    bias_term = *reinterpret_cast<const bool*>(d);
    d += sizeof(bool);

    mWeights = copyToDevice(d + sizeof(int), *reinterpret_cast<const int*>(d));
    d += sizeof(int) + mWeights.count*sizeof(float);

    if (bias_term)
    {
        mBias = copyToDevice(d + sizeof(int), *reinterpret_cast<const int*>(d));
        d += sizeof(int) + mBias.count*sizeof(float);
    }
    else
    {
        mBias.values = nullptr;
    }

    assert(d == a + length);
    //printf("%d %d %d %d %d %d\n", input_channels, input_height, input_width, output_count/(output_height*output_width), output_height, output_width);
}

DepthwiseConvolutionLayer::~DepthwiseConvolutionLayer()
{
    if (bias_term)
    {
        CHECK(cudaFree(const_cast<void*>(mWeights.values)));
        CHECK(cudaFree(const_cast<void*>(mBias.values)));
    }
    else
    {
        CHECK(cudaFree(const_cast<void*>(mWeights.values)));
    }
}

int DepthwiseConvolutionLayer::getNbOutputs() const
{
    return 1;
}

Dims DepthwiseConvolutionLayer::getOutputDimensions(int index, const Dims *inputs, int nbInputDims)
{
    assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);

    return DimsCHW(inputs[0].d[0],
            int(1 + (inputs[0].d[1] + 2 * pad_h_ - (dilation_h_ * (kernel_h_ - 1))) / stride_h_),
            int(1 + (inputs[0].d[2] + 2 * pad_w_ - (dilation_w_ * (kernel_w_ - 1))) / stride_w_));

}

int DepthwiseConvolutionLayer::enqueue(int batchSize, const void * const *inputs, void **outputs, void*, cudaStream_t stream)
{
    DepthwiseConvolutionForward(batchSize*output_count, batchSize, input_channels, output_height, output_width, input_height, input_width,
                                kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, dilation_h_, dilation_w_,
                                reinterpret_cast<const float*>(inputs[0]), reinterpret_cast<float*>(outputs[0]), reinterpret_cast<const float*>(mWeights.values),
            reinterpret_cast<const float*>(mBias.values), bias_term);
    return 0;
}

void DepthwiseConvolutionLayer::configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSzie)
{
    input_channels = inputDims[0].d[0];
    input_height = inputDims[0].d[1];
    input_width = inputDims[0].d[2];

    output_count = outputDims[0].d[0] * outputDims[0].d[1] * outputDims[0].d[2];
    output_height = outputDims[0].d[1];
    output_width = outputDims[0].d[2];
}

size_t DepthwiseConvolutionLayer::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

int DepthwiseConvolutionLayer::initialize()
{
    return 0;
}

void DepthwiseConvolutionLayer::terminate() {}

size_t DepthwiseConvolutionLayer::getSerializationSize()
{
    if(bias_term)
        return sizeof(int) * 15 + sizeof(size_t) + mWeights.count * sizeof(float) + mBias.count * sizeof(float) + sizeof(bool);
    else
        return sizeof(int) * 15 + sizeof(size_t) + mWeights.count * sizeof(float) + sizeof(bool);
}

void DepthwiseConvolutionLayer::serialize(void *buffer)
{
    char* d = reinterpret_cast<char*>(buffer);
    char* a = d;
    *reinterpret_cast<int*>(d) = input_channels;
    d += sizeof(int);
    *reinterpret_cast<int*>(d) = input_height;
    d += sizeof(int);
    *reinterpret_cast<int*>(d) = input_width;
    d += sizeof(int);
    *reinterpret_cast<size_t*>(d) = output_count;
    d += sizeof(size_t);
    *reinterpret_cast<int*>(d) = output_height;
    d += sizeof(int);
    *reinterpret_cast<int*>(d) = output_width;
    d += sizeof(int);
    *reinterpret_cast<int*>(d) = kernel_h_;
    d += sizeof(int);
    *reinterpret_cast<int*>(d) = kernel_w_;
    d += sizeof(int);
    *reinterpret_cast<int*>(d) = pad_h_;
    d += sizeof(int);
    *reinterpret_cast<int*>(d) = pad_w_;
    d += sizeof(int);
    *reinterpret_cast<int*>(d) = stride_h_;
    d += sizeof(int);
    *reinterpret_cast<int*>(d) = stride_w_;
    d += sizeof(int);
    *reinterpret_cast<int*>(d) = dilation_h_;
    d += sizeof(int);
    *reinterpret_cast<int*>(d) = dilation_w_;
    d += sizeof(int);

    *reinterpret_cast<bool*>(d) = bias_term;
    d += sizeof(bool);

    d += copyFromDevice(d, mWeights);

    if(bias_term)
    {
        d += copyFromDevice(d, mBias);
    }

    assert(d == a + getSerializationSize());
}

Weights DepthwiseConvolutionLayer::copyToDevice(const void* hostData, int count)
{
    void* deviceData;
    CHECK(cudaMalloc(&deviceData, count*sizeof(float)));
    CHECK(cudaMemcpy(deviceData, hostData, count*sizeof(float), cudaMemcpyHostToDevice));
    return Weights{DataType::kFLOAT, deviceData, count};
}

int DepthwiseConvolutionLayer::copyFromDevice(char* hostBuffer, Weights deviceWeights)
{
    *reinterpret_cast<int*>(hostBuffer) = deviceWeights.count;
    cudaMemcpy(hostBuffer + sizeof(int), deviceWeights.values, deviceWeights.count*sizeof(float), cudaMemcpyDeviceToHost);
    return sizeof(int) + deviceWeights.count*sizeof(float);
    }
}
