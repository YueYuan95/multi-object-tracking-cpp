#ifndef _KALMAN_TRACKER_H_
#define _KALMAN_TRACKER_H_

#include <iostream>
#include <unistd.h>
#include <memory>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "descriptor.h" 

class KalmanTracker{

    private:
        
        static int trk_count;

        cv::Mat m_measurement;
        cv::KalmanFilter m_kalman_filter;
        cv::Rect_<float> m_bbox;
        /*Descriport should be DeepSort Class not here */
        //Descriptor m_descriptor;
        std::shared_ptr<float> m_descriptor;
        std::vector<cv::Rect_<float>> m_history;

        std::string m_id;
        int m_label;
        int m_state;
        int m_age;
        int m_time_since_update;

    public:

        KalmanTracker(cv::Rect_<float>,int);
        
        /*Get private number*/
        cv::Rect_<float> getBbox();
        std::shared_ptr<float> getDescriptor();
        std::string getId();
        int getLabel();
        int getState();
        int getTime();
        std::vector<cv::Rect_<float>> getHistory();

        /*Set private number*/
        int setState(int);
        int setDescriptor(std::shared_ptr<float>);

        int predict(cv::Mat);
        int update(cv::Mat, cv::Rect_<float>);
        cv::Rect_<float> getRectBox(cv::Mat,float, float, float, float);
}

#endif
