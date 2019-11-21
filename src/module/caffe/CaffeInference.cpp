#include <stdio.h>
#include "CaffeInference.h"

void CaffeInference::readConfig(std::string config_path)
{
    INI::File config_parser(config_path);
    modelFile = config_parser.GetSection("caffe")->GetValue("caffemodel").AsString();
    deployFile = config_parser.GetSection("caffe")->GetValue("prototxt").AsString();
    input_layer_name = config_parser.GetSection("caffe")->GetValue("input_layer_name").AsString();
    output_top_name = getINIStringList(config_parser.GetSection("caffe")->GetValue("output_top_name").AsString());
    input_size = getINIIntList(config_parser.GetSection("pytorch")->GetValue("input_size").AsArray());
    std_list = getINIFloatList(config_parser.GetSection("pytorch")->GetValue("std").AsArray());
    mean_val_list = getINIFloatList(config_parser.GetSection("pytorch")->GetValue("mean_val").AsArray());
    color = config_parser.GetSection("pytorch")->GetValue("color").AsString();
    test_image = config_parser.GetSection("pytorch")->GetValue("test_image").AsString();
}

void CaffeInference::ImageProcess()
{
    cv::Mat img = cv::imread(test_image);
    std::cout << "img size(H*W): " << img.size << std::endl;
    std::cout << "img channels(C): " << img.channels() << std::endl;
    std::cout << "img dims: " << img.dims << std::endl;
    cv::Mat img_RGB;
    cv::resize(img, img, cv::Size(input_size[1], input_size[2]));
    if (color == "RGB")
    {
        cv::cvtColor(img, img_RGB, CV_BGR2RGB);
        img_RGB.convertTo(frame, CV_32F, 1.0);
    }
    else
    {
        img.convertTo(frame, CV_32F, 1.0);
    }
    cv::Mat channels[3];
    cv::split(frame, channels);
    channels[0] -= mean_val_list[0];
    channels[1] -= mean_val_list[1];
    channels[2] -= mean_val_list[2];

    channels[0] /= std_list[0];
    channels[1] /= std_list[1];
    channels[2] /= std_list[2];
    cv::merge(channels, 3, frame);
}

int CaffeInference::get_feature(std::string save_dir)
{
    const size_t m_height = input_size[1];
    const size_t m_width = input_size[2];
    const float m_meanVal = 0;
    const float m_scaleFactor = 1;
    cv::Mat inputblob = cv::dnn::blobFromImage(frame, m_scaleFactor,
                                               cv::Size(m_width, m_height), m_meanVal, false);
    cv::dnn::Net m_net = cv::dnn::readNetFromCaffe(save_dir + "/" + deployFile, save_dir + "/" + modelFile);
    m_net.setInput(inputblob, input_layer_name);
    std::cout << "inputblob size(H*W): " << inputblob.size << std::endl;
    std::cout << "inputblob channels(C): " << inputblob.channels() << std::endl;
    std::cout << "inputblob dims: " << inputblob.dims << std::endl;
    /* 保存图像预处理结果 */
    // std::vector<float> pre_process_res;
    // float *data = (float *)inputblob.data;
    // for (int i = 0; i < 1 * input_size[0] * m_height * m_width; i++)
    // {
    //     pre_process_res.push_back(*(data + i));
    //     std::cout << *(data + i) << std::endl;
    // }
    // std::string save_path = save_dir + "/preprocess.res";
    // FILE *fp = fopen(save_path.c_str(), "w");
    // fwrite(&(pre_process_res[0]), pre_process_res.size(), sizeof(float), fp);
    // fclose(fp);

    std::vector<std::vector<cv::Mat>> out_result;
    m_net.forward(out_result, output_top_name);
    std::vector<cv::Mat> img_result;
    for (int i = 0; i < out_result.size(); i++)
    {
        for (int j = 0; j < out_result[i].size(); j++)
        {
            img_result.push_back(out_result[i][j]);
        }
    }
    std::vector<float> result;
    for (int i = 0; i < img_result.size(); i++)
    {
        float *data = (float *)img_result[i].data;
        int length = 1;
        for (int p = 0; p < img_result[i].size.dims(); p++)
        {
            length = length * img_result[i].size[p];
        }
        for (int j = 0; j < length; j++)
        {
            result.push_back(data[j]);
        }
    }
    std::ofstream outfile;
    outfile.open(save_dir+"/caffemodel.result");
    for(int i=0; i < result.size(); i++){
        outfile << result[i] << std::endl;
    }
    outfile.close();
    /* 以二进制形式保存网络推理结果 */
    // std::string save_path = save_dir + "/caffemodel_14_layer.res";
    // FILE *fp = fopen(save_path.c_str(), "w");
    // fwrite(&(result[0]), result.size(), sizeof(float), fp);
    // fclose(fp);
    std::cout << "The result of caffemodel is saved sucessfully" << std::endl;
    return 0;
}
/* 
int main(int argc,char ** argv)
{
    std::string save_dir = argv[1];
    std::string config_path = save_dir+"/caffe2tensorRT.param";
    CaffeInference CaIn;
    CaIn.readConfig(config_path);
    CaIn.ImageProcess();
    CaIn.get_feature(save_dir);
    return 0;
}
*/