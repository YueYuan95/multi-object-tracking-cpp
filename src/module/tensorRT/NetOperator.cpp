//
// Created by xing on 19-5-1.
//
#include "iniparser.hpp"
#include "NetOperator.h"
#include <unistd.h>
#include "BatchStream.h"
#include "Calibrator.h"
namespace bdavs {
NetOperator::NetOperator() {
    initLibNvInferPlugins(&gLogger, "");
}

NetOperator::~NetOperator()
{
    context->destroy();
    engine->destroy();
    runtime->destroy();

    CHECK(cudaFree(mInputLayer.bottom_data));
    //CHECK(cudaFree(mInputLayer.top_data));
    for(int i=0; i<mOutputLayers.size(); i++)
    {
        //CHECK(cudaFree(mOutputLayers[i].bottom_data));
        CHECK(cudaFree(mOutputLayers[i].top_data));
    }
}
std::vector<float> getINIFloatList(INI::Array ini_array)
{
    std::vector<float> result;
    for (int i =0;i<ini_array.Size();i++)
    {
        result.push_back(float(ini_array[i].AsDouble()));
    }
    return result;
}

std::vector<int> getINIIntList(INI::Array ini_array)
{
    std::vector<int> result;
    for (int i =0;i<ini_array.Size();i++)
    {
        result.push_back(int(ini_array[i].AsInt()));
    }
    return result;
}
std::vector<std::string> getINIStringList(INI::Array ini_array)
{
    std::vector<std::string> result;
    for (int i =0;i<ini_array.Size();i++)
    {
        result.push_back(std::string(ini_array[i].AsString()));
    }
    return result;
}
bool NetOperator::readConfig(std::string model_dir,std::string cofig_name)
{
    // NetParameters para;
    std::string config_path=model_dir+"/"+cofig_name;
    std::cout<<config_path<<std::endl;
    INI::File config_parser(config_path);
    std::string engine_file,model_file,deploy_file,test_image;
    std::string model_name;
    std::string input_layer_name;
    std::vector<std::string> output_layer_name;
    std::vector<int> input_size;
    std::vector<float> std_list;
    std::vector<float> mean_val_list;
    std::string color;
    int batch_size=0;

    engine_file=config_parser.GetSection("caffe")->GetValue("engine_model").AsString();
    model_name=config_parser.GetSection("caffe")->GetValue("model_name").AsString();
    input_layer_name=config_parser.GetSection("caffe")->GetValue("input_layer_name").AsString();
    output_layer_name=getINIStringList(config_parser.GetSection("caffe")->GetValue("output_layer_name").AsArray());
    input_size= getINIIntList(config_parser.GetSection("pytorch")->GetValue("input_size").AsArray());
    std_list=getINIFloatList(config_parser.GetSection("pytorch")->GetValue("std").AsArray());
    mean_val_list=getINIFloatList(config_parser.GetSection("pytorch")->GetValue("mean_val").AsArray());
    color=config_parser.GetSection("pytorch")->GetValue("color").AsString();
    model_file=config_parser.GetSection("caffe")->GetValue("caffemodel").AsString();
    deploy_file=config_parser.GetSection("caffe")->GetValue("prototxt").AsString();
    test_image=config_parser.GetSection("pytorch")->GetValue("test_image").AsString();
    batch_size=config_parser.GetSection("pytorch")->GetValue("batch_size","0").AsInt();

    assert(input_size.size()==3);
    assert(std_list.size()==3);
    assert(mean_val_list.size()==3);

    //for the old version of data
    if (batch_size==0)
    {
        batch_size = 32;
    }
    
    mDims = DimsNCHW{batch_size,input_size[0],input_size[1],input_size[2]};
    mScale = {1/std_list[0],1/std_list[1],1/std_list[2]};
    mMeanVal ={mean_val_list[0],mean_val_list[1],mean_val_list[2]};
    mInputLayerName = input_layer_name;
    mOutputLayerNames = output_layer_name;
    m_color = color;
    m_engine_file=engine_file;
    m_model_file=model_file;
    m_deploy_file=deploy_file;
    m_model_dir = model_dir;
    m_test_image_file=test_image;
    name_=model_name;

// std::cout<<" mDims.h:"<<mDims.h()<<"mDims.w():"<<mDims.w()<<std::endl;
// std::cout<<" mScale:"<<mScale.x<<","<<mScale.y<<","<<mScale.z
//             <<" mMeanVal:"<<mMeanVal.x<<","<<mMeanVal.y<<","<<mMeanVal.z<<std::endl;
}

bool NetOperator::init(NetParameters& para)
{
    mDims = DimsNCHW{para.input_n, para.input_c, para.input_h, para.input_w};
    mScale = {para.scale,para.scale,para.scale};
    mMeanVal = para.mean_val;
    mInputLayerName = para.input_layer_name;
    mOutputLayerNames = para.output_layer_names;

    name_ = para.name;
    // Create inference runtime engine.
    runtime = createInferRuntime(gLogger);
    m_color="BGR";

    if (!runtime)
    {
        printf("[TensorNet] Failed to create inference runtime!\n");
        return false;
    }

    return true;
}
bool NetOperator::loadModel(int gpuID)
{
    std::string model_path=m_model_dir+"/"+m_engine_file;
    std::cout << "model_path is:" << model_path << std::endl;
    return loadModel(model_path.c_str(),gpuID);
}
bool NetOperator::loadModel(const char *modelPath, int gpuID)
{
    // set device
    CHECK(cudaSetDevice(gpuID));
    runtime = createInferRuntime(gLogger);

    if (!runtime)
    {
        printf("[TensorNet] Failed to create inference runtime!\n");
        return false;
    }
    std::stringstream tensorrt_model_stream;
    tensorrt_model_stream.seekg(0, tensorrt_model_stream.beg);

    std::ifstream tensorrt_model_cache_load(modelPath); //model cache to load

    if (!tensorrt_model_cache_load)
    {
        printf("[TensorNet] Failed to open model!\n");
        return false;
    }

    printf("[TensorNet] Cached TensorRT model found, start loading...\n");

    tensorrt_model_stream << tensorrt_model_cache_load.rdbuf();
    tensorrt_model_cache_load.close();

    // support for stringstream deserialization was deprecated in TensorRT v2
    // instead, read the stringstream into a memory buffer and pass that to TRT.
    tensorrt_model_stream.seekg(0, std::ios::end);
    const int modelSize = tensorrt_model_stream.tellg();
    tensorrt_model_stream.seekg(0, std::ios::beg);

#ifdef DEBUG
    printf("[TensorNet] Cached model size : %d\n", modelSize);
#endif

    void* modelMem = malloc(modelSize);

    if (!modelMem)
    {
        printf("[TensorNet] Failed to allocate memory to deserialize model!\n");
        return false;
    }

    tensorrt_model_stream.read((char*)modelMem, modelSize);

    nvinfer1::IPluginFactory *plugin_factory=NULL;
    getPluginAccordingName(name_,plugin_factory);

    engine = runtime->deserializeCudaEngine(modelMem, modelSize, plugin_factory);
    free(modelMem);

    if (!engine)
    {
        printf("[TensorNet] Failed to deserialize CUDA engine!\n");
        return false;
    }

#ifdef DEBUG
    printf("[TensorNet] Deserialize model ok. Number of binding indices %d \n", engine->getNbBindings());
#endif

    tensorrt_model_stream.str("");

    if(!prepareInference())
    {
        printf("[TensorNet] Failed to prepare inference!\n");
        return false;
    }

    return true;
}
bool NetOperator::getPluginAccordingName(std::string name,nvinfer1::IPluginFactory * &plugin_factory)
{
    plugin_factory=NULL;
    if (name=="face_detector_model")
    {
        plugin_factory= &plugin_factory_face_detector;
    }
    if (name=="mark_detector_model")
    {
        plugin_factory= &plugin_factory_mark_detector;
    }
    if (name=="detector_model")
    {
        plugin_factory= &plugin_factory_detector;
    }
    if (name=="vehicle_reid"||name=="pedestrian_reid"||name=="pose"||name=="global_model"||name=="hd_model"||name=="low_model"||name=="up_model"||name=="tracking"||name=="other")
    {
        plugin_factory= &plugin_factory_other;
    }

    if (plugin_factory==NULL)
    {
        std::cout<<"plugin_factory is null\n please set param name"<<std::endl;
        return false;
    }
    return true;
}
bool NetOperator::getPluginAccordingName(std::string name,nvcaffeparser1::IPluginFactory * &plugin_factory)
{
    plugin_factory=NULL;
    std::cout<<"************"<<std::endl;
    std::cout<<name<<std::endl;
    
    if (name=="face_detector_model")
    {
        plugin_factory= &plugin_factory_face_detector;
    }
    if (name=="mark_detector_model")
    {
        plugin_factory= &plugin_factory_mark_detector;
    }
    if (name=="detector_model")
    {
        plugin_factory= &plugin_factory_detector;
    }
    if (name=="vehicle_reid_model"||name=="pedestrian_reid_model"||name=="pose_model"||name=="global_net_model"||name=="hd_net_model"||name=="low_net_model"||name=="up_net_model"||name=="tracking_model"||name=="other")
    {
        plugin_factory= &plugin_factory_other;
    }

    if (plugin_factory==NULL)
    {
        std::cout<<"plugin_factory is null\n please set param name"<<std::endl;
        return false;
    }
    return true;
}
bool NetOperator::prepareInference()
{
    printf("[TensorNet] INPUT_WIDTH: %d , INPUT_HEIGHT: %d \033[0m \n", mDims.w(), mDims.h());

    if (mOutputLayerNames.size() > MAX_OUTPUTS)
    {
        printf("[Cuda]  [TensorNet] failed to set output layer num\n");
        return false;
    }

    /// input
    mInputLayer.name = mInputLayerName;
    mInputLayer.dims = getTensorDims(mInputLayerName.c_str());
    mInputLayer.bottom_data = allocateMemory(mInputLayer.dims, mInputLayerName.c_str());

    // allocate buffer for the image
    //if(!cudaAllocMapped(&mImgCPUPtr, (void**)&mInputLayer.bottom_data, mDims.n()*mDims.c()*mDims.h()*mDims.w()*sizeof(float)))
    //{
    //    printf("[Cuda]  [TensorNet] failed to allocated %zu bytes for image\n", mDims.n()*mDims.h()*mDims.w()*mDims.c()*sizeof(float));
    //    return false;
    //}

    mBuffers[0] = mInputLayer.bottom_data;

    /// output
    for (size_t i = 0; i < mOutputLayerNames.size(); i++)
    {
        Layer outputLayer;
        outputLayer.name = mOutputLayerNames[i];
        outputLayer.dims = getTensorDims(mOutputLayerNames[i].c_str());
        outputLayer.top_data = allocateMemory(outputLayer.dims, mOutputLayerNames[i].c_str());
        mOutputLayers.push_back(outputLayer);

        mBuffers[i + 1] = outputLayer.top_data;
    }

    context = engine->createExecutionContext();

    return true;
}

bool NetOperator::convertCaffeModel(const char *deployFile, const char *modelFile, const char *outFile,std::string mode,std::string data_dir) {
    printf("\033[31;32m%s %s\033[0m\n", deployFile, modelFile);
    if (access(deployFile, 0) == -1 || access(modelFile, 0) == -1)
        return false;

    // create a GIE model from the caffe model and serialize it to a stream
    IHostMemory* gieModelStream = nullptr;
    nvcaffeparser1::IPluginFactory *plugin_factory=NULL;
    getPluginAccordingName(name_,plugin_factory);
    std::cout<<"mDims:"<<mDims.n()<<std::endl;
    caffeToGIEModel(deployFile, modelFile, mOutputLayerNames, mDims.n(), plugin_factory, &gieModelStream,mode,data_dir);

    // cache the trt model
    std::ofstream trtModelFile(outFile);

    trtModelFile.write((char*)gieModelStream->data(), gieModelStream->size());

    printf("[TensorNet] Convert model to tensor model cache : %s completed.\n", outFile);

    trtModelFile.close();

    gieModelStream->destroy();

    return true;
}

void NetOperator::caffeToGIEModel(const char* deployFile,                      // name for caffe prototxt
                                  const char* modelFile,                       // name for model
                                  const std::vector<std::string>& outputLayerNames,   // network outputs
                                  unsigned int maxBatchSize,                          // batch size - NB must be at least as large as the batch we want to run with)
                                  nvcaffeparser1::IPluginFactory* pluginFactory,      // factory for plugin layers
                                  IHostMemory **gieModelStream,                         // output stream for the GIE model
                                  std::string mode,
                                  std::string data_dir) {
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);
    std::cout<<"1"<<std::endl;
    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    std::cout<<"2"<<std::endl;
    ICaffeParser* parser = createCaffeParser();
    std::cout<<"3"<<std::endl;
    parser->setPluginFactory(pluginFactory);
    std::cout<<"4"<<std::endl;
    bool useFp16 = builder->platformHasFastFp16();
    std::cout<<"useFp16:"<<useFp16<<std::endl;
    std::cout<<"5"<<std::endl;
    DataType data_type ;
    if (mode == "int8")
        data_type =DataType::kINT8;
    else
        data_type = DataType::kHALF;
    const IBlobNameToTensor* blobNameToTensor = parser->parse(
                deployFile, modelFile, *network,
                DataType::kHALF );
    std::cout<<"6"<<std::endl;
    BatchStream* calibrationStream{nullptr};
    Int8EntropyCalibrator* calibrator{nullptr};
    if (mode=="int8")
    {
        calibrationStream = new BatchStream(data_dir, 32, 32);
        calibrator = new Int8EntropyCalibrator(*calibrationStream,0 );
    }
    // specify which tensors are outputs
    for (auto& s : outputLayerNames){
        std::cout<<s<<std::endl;
        //std::cout<<(*blobNameToTensor->find(s.c_str()))<<std::endl;;
        *blobNameToTensor->find(s.c_str());
        std::cout<<s<<std::endl;
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }
std::cout<<"7"<<std::endl;
    // build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(512 << 20);
    if (mode!="int8") {
        builder->setFp16Mode(useFp16);
    }
    else{
        builder->setInt8Mode(true);
        builder->setInt8Calibrator(calibrator);
        builder->setDebugSync(true);
        builder->setAverageFindIterations(1);
        builder->setMinFindIterations(1);
    }
std::cout<<"8"<<std::endl;
    ICudaEngine* _engine = builder->buildCudaEngine(*network);
std::cout<<"9"<<std::endl;
    assert(_engine);
std::cout<<"10"<<std::endl;
    //destroy the network and the parser
    network->destroy();
    parser->destroy();
std::cout<<"11"<<std::endl;
    // serialize the engine
    (*gieModelStream) = _engine->serialize();

    _engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
}
DimsCHW NetOperator::getTensorDims(const char* name)
{
    for (int b = 0; b < engine->getNbBindings(); b++)
    {
        if (!strcmp(name, engine->getBindingName(b)))
        {
            //DimsCHW dims = static_cast<DimsCHW&&>(engine->getBindingDimensions(b));
            //std::cout<< name <<" C: " << dims.c() << " H: " << dims.h() << " W: " << dims.w() << std::endl;
            return static_cast<DimsCHW&&>(engine->getBindingDimensions(b));
        }
    }

    return DimsCHW{ 0, 0, 0 };
}

float* NetOperator::allocateMemory(DimsCHW dims, const char* name)
{
    float* ptr;

    size_t size = mDims.n() * dims.c() * dims.h() * dims.w();

//#ifdef DEBUG
    printf("[TensorNet] Allocate memory [%s] size = %d  dims.c() = %d ,dims.h() = %d ,dims.w() = %d\r\n",
           name, (int)size, dims.c(), dims.h(), dims.w());
//#endif

    CHECK(cudaMallocManaged(&ptr, size * sizeof(float)));

    return ptr;
}

bool NetOperator::preprocess_gpu(const cv::Mat& imgBGR)
{
    const int inputChannels = imgBGR.channels();
    const int inputHeight = imgBGR.rows;
    const int inputWidth = imgBGR.cols;
    int color_type=0;
    if (m_color=="RGB")
    {
        color_type=1;
    }
    // std::cout<<"color_type:"<<color_type<<std::endl;
    CHECK(cudaPreImageScaleMean(imgBGR.data, inputHeight, inputWidth, inputChannels,
                                mInputLayer.bottom_data, mDims.h(), mDims.w(), mScale, mMeanVal,color_type));

    return true;
}

bool NetOperator::preprocess_gpu(const std::vector<cv::Mat>& imgBGRs)
{
    float* o_ptr = mInputLayer.bottom_data;

    int offset = mDims.c()*mDims.h()*mDims.w();

    for (size_t i=0; i<imgBGRs.size(); i++)
    {
        const int inputChannels = imgBGRs[i].channels();
        std::cout << "inputChannels is:" << inputChannels << std::endl;
        const int inputHeight = imgBGRs[i].rows;
        std::cout << "inputHeight is:" << inputHeight << std::endl;
        const int inputWidth = imgBGRs[i].cols;
        std::cout << "inputWidth is:" << inputWidth << std::endl;
        int color_type=0;
        if (m_color=="RGB")
        {
            color_type=1;
        }
            // std::cout<<"color_type:"<<color_type<<std::endl;
            // std::cout<<"inputHeight:"<<inputHeight<<" inputWidth:"<<inputWidth<<" inputChannels:"<<inputChannels<<" mDims.h:"<<mDims.h()<<"mDims.w():"<<mDims.w()<<" mScale:"<<mScale.x<<","<<mScale.y<<","<<mScale.z
            // <<" mMeanVal:"<<mMeanVal.x<<","<<mMeanVal.y<<","<<mMeanVal.z<<std::endl;
        CHECK(cudaPreImageScaleMean(imgBGRs[i].data, inputHeight, inputWidth, inputChannels,
                                    o_ptr, mDims.h(), mDims.w(), mScale, mMeanVal,color_type));

        o_ptr += offset;
    }

    return true;
}

bool NetOperator::preprocess_gpu(const AVSGPUMat& imgBGRA)
{
    const int inputChannels = imgBGRA.channels;
    const int inputHeight = imgBGRA.height;
    const int inputWidth = imgBGRA.width;
    int color_type=0;
    if (m_color=="RGB")
    {
        color_type=1;
    }
        // std::cout<<"color_type:"<<color_type<<std::endl;

    CHECK(cudaPreImageScaleMeanV2(imgBGRA.data, inputHeight, inputWidth, inputChannels,
                                  mInputLayer.bottom_data, mDims.h(), mDims.w(), mScale, mMeanVal,color_type));

    return true;
}

bool NetOperator::preprocess_gpu(const std::vector<AVSGPUMat>& imgBGRAs)
{
    float* o_ptr = mInputLayer.bottom_data;

    int offset = mDims.c()*mDims.h()*mDims.w();
    // std::cout<<"Preprocess..........."<<std::endl;
    // std::cout<<"Images Size is "<< imgBGRAs.size()<<std::endl;
    for (int i=0; i<imgBGRAs.size(); i++)
    {
        const int inputChannels = imgBGRAs[i].channels;
        const int inputHeight = imgBGRAs[i].height;
        const int inputWidth = imgBGRAs[i].width;
        int color_type=0;
        if (m_color=="RGB")
        {
            color_type=1;
        }
            // std::cout<<"color_type:"<<color_type<<std::endl;
        //     std::cout<<"index:"<<i<<std::endl;
        // std::cout<<"Image width :"<<imgBGRAs[i].width << " Image height :"<<imgBGRAs[i].height<<" Image Channels is :"<<imgBGRAs[i].channels << \
        //     " MDims.h is :"<< mDims.h() << " MDims.w is :" << mDims.w()<<std::endl;
        //std::cout<<imgBGRAs[i].data<<std::endl;
        CHECK(cudaPreImageScaleMeanV2(imgBGRAs[i].data, inputHeight, inputWidth, inputChannels,
                                      o_ptr, mDims.h(), mDims.w(), mScale, mMeanVal,color_type));
        o_ptr += offset;
    }
    // std::cout<<"Preprocess Done!"<<std::endl;
    return true;
}

bool NetOperator::getHandleImages(const std::vector<cv::Mat>& imgBGRs, std::vector<std::vector<cv::Mat> >&image_list)
{
    image_list.clear();
    std::vector<cv::Mat> temp_images;
    for (int i=0;i<imgBGRs.size();i++)
    {
        temp_images.push_back(imgBGRs[i]);
        if ((i+1)%mDims.n()==0||i==imgBGRs.size()-1){
            image_list.push_back(temp_images);
            temp_images.clear();
        }
    }
    return true;
}
bool NetOperator::getHandleImages(const std::vector<AVSGPUMat>& imgBGRs, std::vector<std::vector<AVSGPUMat> >&image_list)
{
    image_list.clear();
    std::vector<AVSGPUMat> temp_images;
    for (int i=0;i<imgBGRs.size();i++)
    {
        temp_images.push_back(imgBGRs[i]);
        if ((i+1)%mDims.n()==0||i==imgBGRs.size()-1){
            image_list.push_back(temp_images);
            temp_images.clear();
        }
    }
    return true;
}
}



