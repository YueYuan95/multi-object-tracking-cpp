#ifndef _KALMAN_TRACKER_V2_H_
#define _KALMAN_TRACKER_V2_H_

#include <iostream>
#include <unistd.h>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>

class KalmanTrackerV2{

     private:
        int ndim;
        int dt;

        cv::Mat motion_mat;
        cv::Mat update_mat;

        float std_weight_position = 1. / 20;
        float std_weight_velocity = 1. / 160;

     public:

        KalmanTrackerV2();
        std::vector<cv::Mat> initiate(cv::Rect_<float>);
        void predict(cv::Mat&, cv::Mat&);
        void project(cv::Mat, cv::Mat, cv::Mat&, cv::Mat&);
        void update(cv::Mat&, cv::Mat&, cv::Rect_<float>);
        double gating_distance(cv::Mat, cv::Mat, cv::Rect_<float>, int only_position=0);
};
#endif