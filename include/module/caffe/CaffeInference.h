#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/types_c.h>

#include "iniparser.hpp"



class CaffeInference
{
    public:
        void readConfig(std::string config_path);
        void ImageProcess();
        int get_feature(std::string save_dir);

    private:
        std::vector<int> getINIIntList(INI::Array ini_array);
        std::vector<float> getINIFloatList(INI::Array ini_array);
        std::vector<cv::String> getINIStringList(INI::Array ini_array);
        cv::Mat frame;
        std::string modelFile;
        std::string deployFile;
        std::string input_layer_name;
        std::vector<cv::String> output_top_name;
        std::vector<int> input_size;
        std::vector<float> std_list;
        std::vector<float> mean_val_list;
        std::string color;
        std::string test_image;
};
std::vector<float> CaffeInference::getINIFloatList(INI::Array ini_array)
{
    std::vector<float> result;
    for (int i =0;i<ini_array.Size();i++)
    {
        result.push_back(float(ini_array[i].AsDouble()));
    }
    return result;
}

std::vector<int> CaffeInference::getINIIntList(INI::Array ini_array)
{
    std::vector<int> result;
    for (int i =0;i<ini_array.Size();i++)
    {
        result.push_back(int(ini_array[i].AsInt()));
    }
    return result;
}
std::vector<cv::String> CaffeInference::getINIStringList(INI::Array ini_array)
{
    std::vector<cv::String> result;
    for (int i =0;i<ini_array.Size();i++)
    {
        result.push_back(cv::String(ini_array[i].AsString()));
    }
    return result;
}
