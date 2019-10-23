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
        int state;
        int miss_time;
        bool is_activate;
        cv::KalmanFilter m_kalman_filter;
        cv::Rect_<float> m_box;

    public:
        int isActivate();
        int predict();
        int update();

        int getState();
        int getMissTime();
        int re_activate();

        int mark_lost();
        int mark_removed();


}

};

#endif