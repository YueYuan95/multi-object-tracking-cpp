#ifndef __BASE_TRACKER_H__
#define __BASE_TRACKER_H__
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace sort{

#define USE_DEEP 0
#define MAX_MISS_TIME 30

#define STATE_NEW 0
#define STATE_TRACKED 1
#define STATE_LOST 2
#define STATE_REMOVE 3

class Tracker{

    private:
        static int id;
        int m_id;
        int m_state;
        int m_time_since_update;
        int m_label;
        bool m_activate;
        cv::KalmanFilter m_kalman_filter;
        cv::Rect_<float> m_box;

    public:
        bool is_activate();
        int predict();
        int update();

        int get_state();
        int get_label();
        int get_miss_time();
        cv::Rect_<float> get_box();

        int mark_lost();
        int mark_removed();


}

};

#endif