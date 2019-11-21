
#include "extractor.h"
namespace bdavs {
bool Extractor::handleResult(std::vector<Layer> &outputLayers,std::vector<std::vector<BlobData>>& result_blob_data_list,int image_number)
{
    result_blob_data_list.clear();
    result_blob_data_list.resize(image_number);
    // std::cout<<"outputLayers:"<<outputLayers.size()<<std::endl;
    
    for (size_t k=0;k<outputLayers.size();k++) {
        float *output_data = outputLayers[k].top_data;
        for (size_t n = 0; n < image_number; n++) {
            BlobData blob_data;
            blob_data.name = outputLayers[k].name;
            for (int i=0;i<outputLayers[k].dims.c()*outputLayers[k].dims.h()*outputLayers[k].dims.w();i++)
            {
                blob_data.data.push_back(output_data[i]);
            }
            output_data+= mOutputLayers[k].dims.c()*outputLayers[k].dims.h()*outputLayers[k].dims.w();
            result_blob_data_list[n].push_back(blob_data);
        }

    }
    return true;
}

bool Extractor::inference(const std::vector<cv::Mat>& imgBGRs, std::vector<std::vector<BlobData>>& result_blob_data_list)
{
    std::vector<std::vector<cv::Mat> > image_list;
    getHandleImages(imgBGRs,image_list);
    // preprocess
    result_blob_data_list.clear();
    for (int k=0;k<image_list.size();k++)
    {
        // preprocess
        if(!preprocess_gpu(image_list[k]))
        {
            printf("[TensorNet] preprocess Image Failed\n");
            return false;
        }

        // inference
        context->execute(image_list[k].size(), mBuffers);
        std::vector<std::vector<BlobData>> temp_blob_data;
        handleResult(mOutputLayers,temp_blob_data,image_list[k].size());
        for(int i=0;i<temp_blob_data.size();i++)
        {
            result_blob_data_list.push_back(temp_blob_data[i]);
        }   
    }
    assert(result_blob_data_list.size()==imgBGRs.size());
    return true;
}
bool Extractor::inference(const cv::Mat &imgBGR,std::vector<float> &result_blob)
{
    std::vector<cv::Mat> image_list;
    image_list.push_back(imgBGR);
    std::vector<std::vector<BlobData>> result_blob_data_list;
    inference(image_list,result_blob_data_list);
    for (int k=0;k<result_blob_data_list[0].size();k++)
    {
        for (int i=0;i<result_blob_data_list[0][k].data.size();i++)
        {
            result_blob.push_back(result_blob_data_list[0][k].data[i]);
        }
    }
    return true;
}
bool Extractor::inference(const std::vector<AVSGPUMat> &imgBGRAs, std::vector<std::vector<BlobData>> &result_blob_data_list)
{

    std::vector<std::vector<AVSGPUMat> > image_list;
    getHandleImages(imgBGRAs,image_list);
    // preprocess
    result_blob_data_list.clear();

    for (int k=0;k<image_list.size();k++)
    {
        // preprocess
        if(!preprocess_gpu(image_list[k]))
        {
            printf("[TensorNet] preprocess Image Failed\n");
            return false;
        }

        // inference
        context->execute(image_list[k].size(), mBuffers);
        std::vector<std::vector<BlobData>> temp_blob_data;
        handleResult(mOutputLayers,temp_blob_data,image_list[k].size());
        for(int i=0;i<temp_blob_data.size();i++)
        {
            result_blob_data_list.push_back(temp_blob_data[i]);
        }   
    }
    assert(result_blob_data_list.size()==imgBGRAs.size());
    return true;
}
}
