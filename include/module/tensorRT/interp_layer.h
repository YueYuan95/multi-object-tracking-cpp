#ifndef _INTERP_LAYER_H_
#define _INTERP_LAYER_H_

#include <cassert>
#include <NvInferPlugin.h>

using namespace nvinfer1;
using namespace plugin;
namespace bdavs {
void caffe_gpu_interp2(const int channels,
                       const float* data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
                             float* data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2);

template<int height_out_,int width_out_>
class InterpLayer : public IPlugin {
public:
    InterpLayer() {}

    InterpLayer(const void* buffer, size_t length) {
        const char* d = reinterpret_cast<const char*>(buffer);
        const char* a = d;
        channels_ = *reinterpret_cast<const int*>(d);
        d += sizeof(int);
        height_in_ = *reinterpret_cast<const int*>(d);
        d += sizeof(int);
        width_in_ = *reinterpret_cast<const int*>(d);
        d += sizeof(int);
        assert(d == a + length);
    }

    ~InterpLayer() {

    }

    int getNbOutputs() const override {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
        assert(nbInputDims == 1 && index == 0 && inputs[index].nbDims == 3);
        
        return DimsCHW(inputs[0].d[0], height_out_, width_out_);
    }

    int initialize() override {
        return 0;
    }

    void terminate() override {

    }

    size_t getWorkspaceSize(int maxBatchSize) const override{
        return 0;
    }

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override {
        caffe_gpu_interp2(batchSize* channels_,
                          reinterpret_cast<const float*>(inputs[0]),  0, 0, height_in_, width_in_, height_in_ , width_in_,
                               reinterpret_cast<float*>(outputs[0]),  0, 0, height_out_ , width_out_ , height_out_ , width_out_);

        return 0;
    }

    void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSzie) override {
        channels_ = inputDims[0].d[0];
        height_in_ = inputDims[0].d[1];
        width_in_ = inputDims[0].d[2];
    }

    size_t getSerializationSize() override {
        return sizeof(int) * 3;
    }

    void serialize(void *buffer) override {
        char* d = reinterpret_cast<char*>(buffer);
        char* a = d;
        *reinterpret_cast<int*>(d) = channels_;
        d += sizeof(int);
        *reinterpret_cast<int*>(d) = height_in_;
        d += sizeof(int);
        *reinterpret_cast<int*>(d) = width_in_;
        d += sizeof(int);
        assert(d == a + getSerializationSize());
    }

private:
    int channels_;
    int height_in_, width_in_;
    // int height_out_, width_out_;
};
}
#endif //_SOFTMAX_LAYER_H_
