//
// Created by xing on 19-5-1.
//

#ifndef _NET_OPERATOR_H_
#define _NET_OPERATOR_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <NvCaffeParser.h>
#include <NvInferPlugin.h>
#include <fstream>
#include <cuda.h>

//#include "PluginFactory.h"
#include "plugin_factory_detector.h"
#include "plugin_factory_face_detector.h"
#include "plugin_factory_mark_detector.h"
#include "plugin_factory_other.h"
#include "Common.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
namespace bdavs {
#define MAX_OUTPUTS 10

typedef struct
{
    unsigned char* data = nullptr;
    int height = 0;
    int width = 0;
    int channels = 0;
} AVSGPUMat;

typedef struct
{
    int input_w;
    int input_h;
    int input_c;
    int input_n;

    float scale;
    float3 mean_val;
    std::string name;
    std::string input_layer_name;
    std::vector<std::string> output_layer_names;
    std::string color;

} NetParameters;

typedef struct
{
    std::string name;
    DimsCHW dims;
    float* bottom_data{nullptr};
    float* top_data{nullptr};
} Layer;

static Logger gLogger;

cudaError_t cudaPreImageMean(float* output, size_t width, size_t height, float scale, float3& mean_value);

// cudaPreImageNetMean
cudaError_t cudaPreImageScaleMean(unsigned char* input, int inputHeight, int inputWidth, int inputChannels,
                                  float* output, int outputHeight, int outputWidth, float3 scale, const float3& mean_value,const int color_type);

cudaError_t cudaPreImageScaleMeanV2(unsigned char* input, int inputHeight, int inputWidth, int inputChannels,
                                    float* output, int outputHeight, int outputWidth, float3 scale, const float3& mean_value,const int color_type);
cudaError_t cudaCropImage(const  unsigned char* input, int inputWidth, int inputHeight, int inputChannels,
         unsigned char* output, int x1, int y1, int x2, int y2);

class NetOperator
{
public:
    NetOperator();
    ~NetOperator();
    bool readConfig(std::string model_dir,std::string cofig_name);
    bool getPluginAccordingName(std::string name,nvinfer1::IPluginFactory * &plugin_factory);
    bool getPluginAccordingName(std::string name,nvcaffeparser1::IPluginFactory * &plugin_factory);
    bool convertCaffeModel(const char *deployFile, const char *modelFile, const char *outFile,std::string mode="",std::string image_dir="");
    void caffeToGIEModel(const char* deployFile,                      // name for caffe prototxt
                                  const char* modelFile,                       // name for model
                                  const std::vector<std::string>& outputLayerNames,   // network outputs
                                  unsigned int maxBatchSize,                          // batch size - NB must be at least as large as the batch we want to run with)
                                  nvcaffeparser1::IPluginFactory* pluginFactory,      // factory for plugin layers
                                  IHostMemory **gieModelStream,
                                  std::string mode,
                                  std::string data_dir);
    bool init(NetParameters& para);
    bool loadModel(int gpuID=0);
    bool loadModel(const char* modelPath, int gpuID=0);

    bool getHandleImages(const std::vector<cv::Mat>& imgBGRs, std::vector<std::vector<cv::Mat> >&image_list);
    bool getHandleImages(const std::vector<AVSGPUMat>& imgBGRs, std::vector<std::vector<AVSGPUMat> >&image_list);
    // cpu to gpu mode
    bool preprocess_gpu(const std::vector<cv::Mat>& imgBGRs);
    bool preprocess_gpu(const cv::Mat& imgBGR);

    // gpu to gou mode
    bool preprocess_gpu(const std::vector<AVSGPUMat>& imgBGRs);
    bool preprocess_gpu(const AVSGPUMat& imgBGR);

private:
    bool prepareInference();

    DimsCHW getTensorDims(const char* name);

    float* allocateMemory(DimsCHW dims, const char* name);

public:
    IExecutionContext* context{nullptr};

    DimsNCHW mDims;

    std::vector<Layer> mOutputLayers;

    void* mBuffers[MAX_OUTPUTS+1];

    std::string m_engine_file,m_model_file,m_deploy_file,m_model_dir,m_test_image_file;
private:
    ICudaEngine* engine{nullptr};

    IRuntime* runtime{nullptr};

//    PluginFactory pluginFactory;
    PluginFactoryFaceDetector plugin_factory_face_detector;
    PluginFactoryMarkDetector plugin_factory_mark_detector;
    PluginFactoryDetector plugin_factory_detector;
    PluginFactoryOther plugin_factory_other;

    Layer mInputLayer;
    std::string m_color;
    float3 mScale;
    float3 mMeanVal;
    std::string name_;
    std::string mInputLayerName;
    std::vector<std::string> mOutputLayerNames;

    void* mImgCPUPtr{nullptr};
    void* mImgGPUPtr{nullptr};

};
}
#endif //_NET_OPERATOR_H_
