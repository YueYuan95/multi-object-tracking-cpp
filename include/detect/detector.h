#ifndef _DETECTOR_H_
#define _DETECTOR_H_

#include <iostream>
#include <queue>
#include <fstream>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "byavs.h"


class Detector{

    public:

        std::map<int, std::vector<cv::Rect_<float>>> frame_det_map;
        std::map<int, std::vector<float>> frame_det_score;
        //void to_do(string seq);
        int read_txt(std::string);
        void split_string(const std::string& s, std::vector<std::string>& v, const std::string& c);
        //int inference(int frame, std::vector<cv::Rect_<float>>& destination);
        int inference(int frame, std::vector<cv::Rect_<float>>& destination, std::vector<float>& destination_score);
};

#endif