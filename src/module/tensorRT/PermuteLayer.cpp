
#include "PermuteLayer.h"
namespace bdavs {
PermuteLayer::PermuteLayer(int order0, int order1, int order2, int order3)
{
    assert(order0 == 0);
    order[0] = order0;
    order[1] = order1;
    order[2] = order2;
    order[3] = order3;

    num_axes = 4;
}

PermuteLayer::PermuteLayer(int order0, int order1, int order2)
{
    order[0] = order0;
    order[1] = order1;
    order[2] = order2;

    num_axes = 3;
}

PermuteLayer::PermuteLayer(const void *buffer, size_t length)
{
    const char* d = reinterpret_cast<const char*>(buffer);

    const char* a = d;

    mInputChannels = *reinterpret_cast<const int*>(d);
    d += sizeof(int);

    mInputHeight = *reinterpret_cast<const int*>(d);
    d += sizeof(int);

    mInputWidth = *reinterpret_cast<const int*>(d);
    d += sizeof(int);

    mOutputChannels = *reinterpret_cast<const int*>(d);
    d += sizeof(int);

    mOutputHeight = *reinterpret_cast<const int*>(d);
    d += sizeof(int);

    mOutputWidth = *reinterpret_cast<const int*>(d);
    d += sizeof(int);

    num_axes = *reinterpret_cast<const int*>(d);
    d += sizeof(int);

    for(int i=0; i<num_axes; i++)
    {
        order[i] = *reinterpret_cast<const int*>(d);
        d += sizeof(int);
    }

    assert(d == a + length);
}

PermuteLayer::~PermuteLayer()
{

}

int PermuteLayer::getNbOutputs() const
{
    return 1;
}

Dims PermuteLayer::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(nbInputDims == 1);
    assert(index == 0);
    assert(inputs[index].nbDims == 3);
    assert(num_axes == 4 || num_axes == 3);

    if(num_axes == 4)
        return DimsCHW(inputs[0].d[order[1]-1], inputs[0].d[order[2]-1], inputs[0].d[order[3]-1]);

    if(num_axes == 3)
    {
        assert(inputs[0].d[2] == 1);
        return DimsCHW(inputs[0].d[0], inputs[0].d[1], 1); // NxCx(HW)
    }
}

int PermuteLayer::initialize()
{
    return 0;
}

void PermuteLayer::terminate()
{

}

size_t PermuteLayer::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

int PermuteLayer::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    CHECK(cudaMallocManaged(&permute_order, num_axes*sizeof(int)));
    CHECK(cudaMallocManaged(&new_steps, num_axes*sizeof(int)));
    CHECK(cudaMallocManaged(&old_steps, num_axes*sizeof(int)));

    for(int i=0; i<num_axes; i++)
        permute_order[i] = order[i];

    if(num_axes == 4)
    {
        old_steps[0] = mInputChannels*mInputHeight*mInputWidth;
        old_steps[1] = mInputHeight*mInputWidth;
        old_steps[2] = mInputWidth;
        old_steps[3] = 1;

        new_steps[0] = mOutputChannels*mOutputHeight*mOutputWidth;
        new_steps[1] = mOutputHeight*mOutputWidth;
        new_steps[2] = mOutputWidth;
        new_steps[3] = 1;
    }

    if(num_axes == 3)
    {
        assert(mInputWidth == 1 && mOutputWidth == 1);
        assert(mInputChannels == mOutputChannels);

        if(permute_order[0] == 2 && permute_order[1] == 0 && permute_order[2] == 1)
        {
            old_steps[0] = mInputChannels*(mInputHeight*mInputWidth);
            old_steps[1] = (mInputHeight*mInputWidth);
            old_steps[2] = 1;

            new_steps[0] = batchSize*mOutputChannels;
            new_steps[1] = mOutputChannels;
            new_steps[2] = 1;
        }

        if(permute_order[0] == 1 && permute_order[1] == 0 && permute_order[2] == 2)
        {
            old_steps[0] = mInputChannels*batchSize;
            old_steps[1] = mInputChannels;
            old_steps[2] = 1;

            new_steps[0] = mOutputHeight*mOutputChannels;
            new_steps[1] = mOutputChannels;
            new_steps[2] = 1;
        }

    }

    int count = batchSize*mInputChannels*mInputHeight*mInputWidth;

    bool need_permute = false;
    for(int i=0; i<num_axes; i++)
    {
        if(permute_order[i] != i)
        {
            need_permute = true;
            break;
        }
    }

    PermuteForward(reinterpret_cast<const float*>(inputs[0]), count, permute_order, new_steps, old_steps, num_axes, need_permute,
            reinterpret_cast<float*>(outputs[0]));

    CHECK(cudaFree(permute_order));
    permute_order = nullptr;
    CHECK(cudaFree(new_steps));
    new_steps = nullptr;
    CHECK(cudaFree(old_steps));
    old_steps = nullptr;

    return 0;
}

void PermuteLayer::configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSzie)
{

    mInputChannels = inputDims[0].d[0];
    mInputHeight = inputDims[0].d[1];
    mInputWidth = inputDims[0].d[2];

    mOutputChannels = outputDims[0].d[0];
    mOutputHeight = outputDims[0].d[1];
    mOutputWidth = outputDims[0].d[2];
}

size_t PermuteLayer::getSerializationSize()
{
    return sizeof(int)*(7+num_axes);
}

void PermuteLayer::serialize(void *buffer)
{
    char* d = reinterpret_cast<char*>(buffer);

    char* a = d;

    *reinterpret_cast<int*>(d) = mInputChannels;
    d += sizeof(int);

    *reinterpret_cast<int*>(d) = mInputHeight;
    d += sizeof(int);

    *reinterpret_cast<int*>(d) = mInputWidth;
    d += sizeof(int);

    *reinterpret_cast<int*>(d) = mOutputChannels;
    d += sizeof(int);

    *reinterpret_cast<int*>(d) = mOutputHeight;
    d += sizeof(int);

    *reinterpret_cast<int*>(d) = mOutputWidth;
    d += sizeof(int);

    *reinterpret_cast<int*>(d) = num_axes;
    d += sizeof(int);

    for(int i=0; i<num_axes; i++)
    {
        *reinterpret_cast<int*>(d) = order[i];
        d += sizeof(int);
    }

    assert(d == a + getSerializationSize());
    }
}
