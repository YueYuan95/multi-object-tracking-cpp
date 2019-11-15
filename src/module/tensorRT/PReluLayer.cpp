//
// Created by svt on 19-5-8.
//

#include "PReluLayer.h"
namespace bdavs {

PReluLayer::PReluLayer() {}

PReluLayer::PReluLayer(const void* buffer, size_t length)
{
    const char* d = reinterpret_cast<const char*>(buffer);

    const char* a = d;

    mCount = *reinterpret_cast<const int*>(d);
    d += sizeof(int);

    mChannels = *reinterpret_cast<const int*>(d);
    d += sizeof(int);

    mDim = *reinterpret_cast<const int*>(d);
    d += sizeof(int);

    assert(d == a + length);
}

int PReluLayer::getNbOutputs() const
{
    return 1;
}

Dims PReluLayer::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(nbInputDims == 1);
    assert(index == 0);
    assert(inputs[index].nbDims == 3);

    return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}

int PReluLayer::initialize()
{
    return 0;
}

void PReluLayer::terminate() {}

size_t PReluLayer::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

int PReluLayer::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    //printf("count: %d   channels: %d   dim: %d\n", mCount, mChannels, mDim);
    PReLUForward(mCount*batchSize, mChannels, mDim, (float*)(*inputs), static_cast<float*>(*outputs));
    return 0;
}

void PReluLayer::configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSzie)
{
    mCount = inputDims[0].d[0] * inputDims[0].d[1] * inputDims[0].d[2];

    mChannels = inputDims[0].d[0];

    mDim = inputDims[0].d[1] * inputDims[0].d[2];
}

size_t PReluLayer::getSerializationSize()
{
    return 3*sizeof(int);
}

void PReluLayer::serialize(void *buffer)
{
    char* d = reinterpret_cast<char*>(buffer);

    char* a = d;

    *reinterpret_cast<int*>(d) = mCount;
    d += sizeof(int);

    *reinterpret_cast<int*>(d) = mChannels;
    d += sizeof(int);

    *reinterpret_cast<int*>(d) = mDim;
    d += sizeof(int);

    assert(d == a + getSerializationSize());
}
}