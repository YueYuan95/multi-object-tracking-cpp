
#include "LSTMLayer.h"
namespace bdavs {

LSTMLayer::LSTMLayer(const Weights* weights, int nbWeights, int num_output, int time_step)
{
    assert(nbWeights == 3);

    nbOutputChannels = num_output;
    nbTimeStep = time_step;

    mKernelWeights0 = copyToDevice(weights[0].values, weights[0].count);
    mBiasWeights0 = copyToDevice(weights[1].values, weights[1].count);
    mKernelWeights1 = copyToDevice(weights[2].values, weights[2].count);
}

LSTMLayer::LSTMLayer(const void *buffer, size_t length)
{
    const char* d = reinterpret_cast<const char*>(buffer);

    const char* a = d;

    mInputChannels = *reinterpret_cast<const int*>(d);
    d += sizeof(int);

    mInputHeight = *reinterpret_cast<const int*>(d);
    d += sizeof(int);

    mInputWidth = *reinterpret_cast<const int*>(d);
    d += sizeof(int);

    nbOutputChannels = *reinterpret_cast<const int*>(d);
    d += sizeof(int);

    nbTimeStep = *reinterpret_cast<const int*>(d);
    d += sizeof(int);

    mKernelWeights0 = copyToDevice(d + sizeof(int), *reinterpret_cast<const int*>(d));
    d += sizeof(int) + mKernelWeights0.count*sizeof(float);

    mBiasWeights0 = copyToDevice(d + sizeof(int), *reinterpret_cast<const int*>(d));
    d += sizeof(int) + mBiasWeights0.count*sizeof(float);

    mKernelWeights1 = copyToDevice(d + sizeof(int), *reinterpret_cast<const int*>(d));
    d += sizeof(int) + mKernelWeights1.count*sizeof(float);

    assert(d == a + length);
}

LSTMLayer::~LSTMLayer()
{
    CHECK(cudaFree(const_cast<void*>(mKernelWeights0.values)));
    mKernelWeights0.values = nullptr;
    CHECK(cudaFree(const_cast<void*>(mBiasWeights0.values)));
    mBiasWeights0.values = nullptr;
    CHECK(cudaFree(const_cast<void*>(mKernelWeights1.values)));
    mKernelWeights1.values = nullptr;
}

int LSTMLayer::getNbOutputs() const
{
    return 1;
}

Dims LSTMLayer::getOutputDimensions(int index, const Dims *inputs, int nbInputDims)
{
    assert(nbInputDims == 1);
    assert(index == 0);
    assert(inputs[index].nbDims == 3);
    assert(inputs[index].d[2] == 1);

    return DimsCHW(nbOutputChannels, inputs[0].d[1], 1);
}

int LSTMLayer::initialize()
{
    CUDNNCHECK(cublasCreate(&mCublas));

    return 0;
}

void LSTMLayer::terminate()
{
    CUDNNCHECK(cublasDestroy(mCublas));
}

size_t LSTMLayer::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

int LSTMLayer::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    // ------------------------ step1 ------------------------------------------
    //  step 1.1 -> slice -> T N -> {1 N, ...}
    float* cont;
    CHECK(cudaMalloc<float>(&cont, batchSize*nbTimeStep*sizeof(float)));
    CHECK(cudaMemset(cont, 0.0f, batchSize*sizeof(float)));
    CHECK(cudaMemset(cont + batchSize, 1.0f, (nbTimeStep-1)*batchSize*sizeof(float)));

    std::vector<float*> cont_s;
    cont_s.resize(nbTimeStep);
    for(int i=0; i<nbTimeStep; i++)
    {
        CHECK(cudaMalloc<float>(&cont_s[i], batchSize*sizeof(float)));
    }

    int num_slices, slice_size, bottom_slice_axis, top_slice_axis;

    num_slices = 1;
    slice_size = batchSize;
    bottom_slice_axis = nbTimeStep;
    top_slice_axis = 1;

    SliceForward(reinterpret_cast<const float*>(cont), num_slices, slice_size, bottom_slice_axis, top_slice_axis, cont_s);

    // step 1.2 -> fc
    float* W_xc_x;
    CHECK(cudaMalloc<float>(&W_xc_x, nbTimeStep*batchSize*(nbOutputChannels*4)*sizeof(float)));

    int N, M, K, lda, ldb;

    N = nbOutputChannels*4;
    M = nbTimeStep*batchSize;
    K = mInputChannels;  //
    lda = K;
    ldb = K;

    constexpr float kONE = 1.0f, kZERO = 0.0f;
    // Do matrix multiplication.
    cublasSetStream(mCublas, stream);
    CUDNNCHECK(cublasSgemm(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &kONE,
                           reinterpret_cast<const float*>(mKernelWeights0.values), ldb,
                           reinterpret_cast<const float*>(inputs[0]), lda, &kZERO,
               W_xc_x, N));

    // bias
    N = nbOutputChannels*4;
    M = nbTimeStep*batchSize;
    K = 1;  //
    lda = K;
    ldb = N;

    void* bias_multiplier;
    CHECK(cudaMalloc(&bias_multiplier, M*sizeof(float)));
    CHECK(cudaMemset(bias_multiplier, 1.0f, M*sizeof(float)));
    CUDNNCHECK(cublasSgemm(mCublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &kONE,
                           reinterpret_cast<const float*>(mBiasWeights0.values), ldb,
                           reinterpret_cast<const float*>(bias_multiplier), lda, &kONE,
                           W_xc_x, N));

    // step 1.3 -> slice -> (hw) n c -> {1 n c, ...}
    std::vector<float*> W_xc_x_s;
    W_xc_x_s.resize(nbTimeStep);
    for(int i=0; i<nbTimeStep; i++)
    {
        CHECK(cudaMalloc<float>(&W_xc_x_s[i], batchSize*nbOutputChannels*4*sizeof(float)));
    }

    num_slices = 1;
    slice_size = batchSize*nbOutputChannels*4;
    bottom_slice_axis = nbTimeStep;
    top_slice_axis = 1;

    SliceForward(reinterpret_cast<const float*>(W_xc_x), num_slices, slice_size, bottom_slice_axis, top_slice_axis, W_xc_x_s);

    // ------------------------ step2 ------------------------------------------
    // step 2.1  - >  scale
    // input
    float* input;
    CHECK(cudaMalloc<float>(&input, batchSize*nbOutputChannels*sizeof(float)));
    CHECK(cudaMemset(input, 0, batchSize*nbOutputChannels*sizeof(float)));

    // scale output
    std::vector<float*> h_conted_s;
    h_conted_s.resize(nbTimeStep);
    for (int i=0; i<nbTimeStep; i++)
    {
        CHECK(cudaMalloc<float>(&h_conted_s[i], batchSize*nbOutputChannels*sizeof(float)));
    }

    // fc ouput
    std::vector<float*> W_hc_h_s;
    W_hc_h_s.resize(nbTimeStep);
    for (int i=0; i<nbTimeStep; i++)
    {
        CHECK(cudaMalloc<float>(&W_hc_h_s[i], batchSize*(nbOutputChannels*4)*sizeof(float)));
    }

    // eltwise ouput
    std::vector<float*> gate_input_s;
    gate_input_s.resize(nbTimeStep);
    for (int i=0; i<nbTimeStep; i++)
    {
        CHECK(cudaMalloc<float>(&gate_input_s[i], batchSize*(nbOutputChannels*4)*sizeof(float)));
    }

    // lstm_unit out
    std::vector<float*> X_acts_s, C_s, H_s;
    X_acts_s.resize(nbTimeStep);
    C_s.resize(nbTimeStep);
    H_s.resize(nbTimeStep);
    for (int i=0; i<nbTimeStep; i++)
    {
        CHECK(cudaMalloc<float>(&X_acts_s[i], batchSize*nbOutputChannels*4*sizeof(float)));
        CHECK(cudaMalloc<float>(&C_s[i], batchSize*nbOutputChannels*sizeof(float)));
        CHECK(cudaMalloc<float>(&H_s[i], batchSize*nbOutputChannels*sizeof(float)));
    }

    int inner_dim, scale_dim, count, X_count;
    inner_dim = nbOutputChannels;
    scale_dim = batchSize;
    count = batchSize*nbOutputChannels;

    int bottom_concat_axis, top_concat_axis, concat_input_size, num_concats;

    for(int i=0; i<nbTimeStep; i++)
    {
        // step 2.2 -> scale
        ScaleForward(input, cont_s[i], count, scale_dim, inner_dim, h_conted_s[i]);

        // step 2.3 -> fc
        N = nbOutputChannels*4;
        M = batchSize;
        K = nbOutputChannels;  //
        lda = K;
        ldb = K;

        constexpr float kONE = 1.0f, kZERO = 0.0f;
        // Do matrix multiplication.
        //cublasSetStream(mCublas, stream);
        CUDNNCHECK(cublasSgemm(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &kONE,
                               reinterpret_cast<const float*>(mKernelWeights1.values), ldb,
                               reinterpret_cast<const float*>(h_conted_s[i]), lda, &kZERO,
                               W_hc_h_s[i], N));

        // step 2.4 -> eltwise sum  W_hc_h_s[i] + W_xc_x_s[i] -> gate_input_s[i]
        N = batchSize*nbOutputChannels*4;
        CUDNNCHECK(cublasSaxpy(mCublas, N, &kONE, W_hc_h_s[i], 1, gate_input_s[i], 1));
        CUDNNCHECK(cublasSaxpy(mCublas, N, &kONE, W_xc_x_s[i], 1, gate_input_s[i], 1));

        // step 2.5 -> lstm_unit
        X_count = N;
        LSTMUnitForward(count, input, gate_input_s[i], cont_s[i], X_acts_s[i], C_s[i], H_s[i], X_count, nbOutputChannels);
    }

    // ------------------------ step3 ------------------------------------------

    // step 3.1 -> concat H_s[i] -> h
    num_concats = 1;
    concat_input_size = batchSize*nbOutputChannels;
    bottom_concat_axis = 1;
    top_concat_axis = nbTimeStep;
    ConcatForward(H_s, bottom_concat_axis, top_concat_axis, concat_input_size, num_concats,
                  reinterpret_cast<float*>(outputs[0]));

    // ------------------------ step4 ------------------------------------------

    CHECK(cudaFree(input)); input = nullptr;
    CHECK(cudaFree(bias_multiplier)); bias_multiplier = nullptr;
    CHECK(cudaFree(W_xc_x)); W_xc_x = nullptr;
    CHECK(cudaFree(cont)); cont = nullptr;

    for(int i=0; i<nbTimeStep; i++)
    {
        CHECK(cudaFree(cont_s[i])); cont_s[i] = nullptr;
        CHECK(cudaFree(W_xc_x_s[i])); W_xc_x_s[i] = nullptr;
        CHECK(cudaFree(h_conted_s[i])); h_conted_s[i] = nullptr;
        CHECK(cudaFree(W_hc_h_s[i])); W_hc_h_s[i] = nullptr;
        CHECK(cudaFree(gate_input_s[i])); gate_input_s[i] = nullptr;
        CHECK(cudaFree(X_acts_s[i])); X_acts_s[i] = nullptr;
        CHECK(cudaFree(C_s[i])); C_s[i] = nullptr;
        CHECK(cudaFree(H_s[i])); H_s[i] = nullptr;
    }

    return 0;
}

void LSTMLayer::configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSzie)
{
    mInputChannels = inputDims[0].d[0];
    mInputHeight = inputDims[0].d[1]; //hw
    mInputWidth = inputDims[0].d[2];
}

size_t LSTMLayer::getSerializationSize()
{
    return sizeof(int)*8 + mKernelWeights0.count*sizeof(float) + \
            mBiasWeights0.count*sizeof(float) + mKernelWeights1.count*sizeof(float);
}

void LSTMLayer::serialize(void *buffer)
{
    char* d = reinterpret_cast<char*>(buffer);

    char* a = d;

    *reinterpret_cast<int*>(d) = mInputChannels;
    d += sizeof(int);

    *reinterpret_cast<int*>(d) = mInputHeight;
    d += sizeof(int);

    *reinterpret_cast<int*>(d) = mInputWidth;
    d += sizeof(int);

    *reinterpret_cast<int*>(d) = nbOutputChannels;
    d += sizeof(int);

    *reinterpret_cast<int*>(d) = nbTimeStep;
    d += sizeof(int);

    d += copyFromDevice(d, mKernelWeights0);
    d += copyFromDevice(d, mBiasWeights0);
    d += copyFromDevice(d, mKernelWeights1);

    assert(d == a + getSerializationSize());
}

Weights LSTMLayer::copyToDevice(const void* hostData, int count)
{
    void* deviceData;
    CHECK(cudaMalloc(&deviceData, count*sizeof(float)));
    CHECK(cudaMemcpy(deviceData, hostData, count*sizeof(float), cudaMemcpyHostToDevice));
    return Weights{DataType::kFLOAT, deviceData, count};
}

int LSTMLayer::copyFromDevice(char* hostBuffer, Weights deviceWeights)
{
    *reinterpret_cast<int*>(hostBuffer) = deviceWeights.count;
    CHECK(cudaMemcpy(hostBuffer + sizeof(int), deviceWeights.values, deviceWeights.count*sizeof(float), cudaMemcpyDeviceToHost));
    return sizeof(int) + deviceWeights.count*sizeof(float);
}

}