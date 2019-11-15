
#include "InnerProductLayer.h"
namespace bdavs {
InnerProductLayer::InnerProductLayer(const Weights* weights, int nbWeights, int nbOutputChannels, int num_axis)
{
    bias_term = nbWeights == 2 ? true : false;

    if(bias_term)
    {
        mKernelWeights = copyToDevice(weights[0].values, weights[0].count);
        mBiasWeights = copyToDevice(weights[1].values, weights[1].count);
    }
    else
    {
        mKernelWeights = copyToDevice(weights[0].values, weights[0].count);
    }

    mNbOutputChannels = nbOutputChannels;
    mNbNumAxis = num_axis;
}

InnerProductLayer::InnerProductLayer(const void *buffer, size_t length)
{
    const char *d = static_cast<const char*>(buffer);
    const char *a = d;

    mNbInputChannels = *reinterpret_cast<const int*>(d);
    d += sizeof(int);

    mNbInputHeight = *reinterpret_cast<const int*>(d);
    d += sizeof(int);

    mNbInputWidth = *reinterpret_cast<const int*>(d);
    d += sizeof(int);

    mNbOutputChannels = *reinterpret_cast<const int*>(d);
    d += sizeof(int);

    mNbNumAxis = *reinterpret_cast<const int*>(d);
    d += sizeof(int);

    bias_term = *reinterpret_cast<const bool*>(d);
    d += sizeof(bool);

    mKernelWeights = copyToDevice(d + sizeof(int), *reinterpret_cast<const int*>(d));
    d += sizeof(int) + mKernelWeights.count*sizeof(float);

    if (bias_term)
    {
        mBiasWeights = copyToDevice(d + sizeof(int), *reinterpret_cast<const int*>(d));
        d += sizeof(int) + mBiasWeights.count*sizeof(float);
    }
    else
    {
        mBiasWeights.values = nullptr;
    }

    assert(d == a + length);
}

InnerProductLayer::~InnerProductLayer()
{
    if (bias_term)
    {
        CHECK(cudaFree(const_cast<void*>(mKernelWeights.values)));
        mKernelWeights.values = nullptr;
        CHECK(cudaFree(const_cast<void*>(mBiasWeights.values)));
        mBiasWeights.values = nullptr;
    }
    else
    {
        CHECK(cudaFree(const_cast<void*>(mKernelWeights.values)));
        mBiasWeights.values = nullptr;
    }
}

int InnerProductLayer::getNbOutputs() const
{
    return 1;
}

Dims InnerProductLayer::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    if (mNbNumAxis == 3)
    {
        assert(index == 0 && nbInputDims == 1 && inputs[index].nbDims == 3);

        return DimsCHW(mNbOutputChannels, inputs[0].d[0], inputs[0].d[1]);  // NxCxHxW
    }

    if (mNbNumAxis == 2)
    {
        assert(index == 0 && nbInputDims == 1 && inputs[index].nbDims == 3);

        return DimsCHW(mNbOutputChannels, inputs[0].d[1], 1);  // NxCx(HW)x1  (HW)xNxC
    }

}

int InnerProductLayer::initialize()
{
    CUDNNCHECK(cublasCreate(&mCublas));

    return 0;
}

void InnerProductLayer::terminate()
{
    CUDNNCHECK(cublasDestroy(mCublas));
}

size_t InnerProductLayer::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

int InnerProductLayer::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    int N, M, K, lda, ldb;

    N = mNbOutputChannels;
    M = mNbInputHeight*mNbInputWidth*batchSize;
    K = mNbInputChannels;  //
    lda = K;
    ldb = K;

    constexpr float kONE = 1.0f, kZERO = 0.0f;
    // Do matrix multiplication.
    cublasSetStream(mCublas, stream);
    CUDNNCHECK(cublasSgemm(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &kONE,
                           reinterpret_cast<const float*>(mKernelWeights.values), ldb,
                           reinterpret_cast<const float*>(inputs[0]), lda, &kZERO,
               reinterpret_cast<float*>(outputs[0]), N));
    if(bias_term)
    {

        // bias
        N = mNbOutputChannels;
        M = mNbInputHeight*mNbInputWidth*batchSize;
        K = 1;  //
        lda = K;
        ldb = N;

        void* bias_multiplier;

        CHECK(cudaMalloc(&bias_multiplier, M*sizeof(float)));
        CHECK(cudaMemset(bias_multiplier, 1.0f, M*sizeof(float)));

        CUDNNCHECK(cublasSgemm(mCublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &kONE,
                               reinterpret_cast<const float*>(mBiasWeights.values), ldb,
                               reinterpret_cast<const float*>(bias_multiplier), lda, &kONE,
                               reinterpret_cast<float*>(outputs[0]), N));

        CHECK(cudaFree(bias_multiplier));
        bias_multiplier = nullptr;
    }

    return 0;
}

void InnerProductLayer::configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSzie)
{
    if (mNbNumAxis == 3)
    {
        mNbInputChannels = inputDims[0].d[2];
        mNbInputHeight= inputDims[0].d[0];
        mNbInputWidth = inputDims[0].d[1];
    }

    if (mNbNumAxis == 2)
    {
        mNbInputChannels = inputDims[0].d[0];
        mNbInputHeight= inputDims[0].d[1];
        mNbInputWidth = inputDims[0].d[2];
    }
}

size_t InnerProductLayer::getSerializationSize()
{
    if(bias_term)
        return sizeof(int) * 7 + mKernelWeights.count * sizeof(float) + mBiasWeights.count * sizeof(float) + sizeof(bool);
    else
        return sizeof(int) * 6 + mKernelWeights.count * sizeof(float) + sizeof(bool);
}

void InnerProductLayer::serialize(void *buffer)
{
    char* d = reinterpret_cast<char*>(buffer);

    char* a = d;

    *reinterpret_cast<int*>(d) = mNbInputChannels;
    d += sizeof(int);

    *reinterpret_cast<int*>(d) = mNbInputHeight;
    d += sizeof(int);

    *reinterpret_cast<int*>(d) = mNbInputWidth;
    d += sizeof(int);

    *reinterpret_cast<int*>(d) = mNbOutputChannels;
    d += sizeof(int);

    *reinterpret_cast<int*>(d) = mNbNumAxis;
    d += sizeof(int);

    *reinterpret_cast<bool*>(d) = bias_term;
    d += sizeof(bool);

    d += copyFromDevice(d, mKernelWeights);

    if(bias_term)
    {
        d += copyFromDevice(d, mBiasWeights);
    }

    assert(d == a + getSerializationSize());
}

size_t InnerProductLayer::type2size(DataType type)
{
    return type == DataType::kFLOAT ? sizeof(float) : sizeof(__half);
}

Weights InnerProductLayer::copyToDevice(const void* hostData, int count)
{
    void* deviceData;
    CHECK(cudaMalloc(&deviceData, count*sizeof(float)));
    CHECK(cudaMemcpy(deviceData, hostData, count*sizeof(float), cudaMemcpyHostToDevice));
    return Weights{DataType::kFLOAT, deviceData, count};
}

int InnerProductLayer::copyFromDevice(char* hostBuffer, Weights deviceWeights)
{
    *reinterpret_cast<int*>(hostBuffer) = deviceWeights.count;
    cudaMemcpy(hostBuffer + sizeof(int), deviceWeights.values, deviceWeights.count*sizeof(float), cudaMemcpyDeviceToHost);
    return sizeof(int) + deviceWeights.count*sizeof(float);
    }
}
