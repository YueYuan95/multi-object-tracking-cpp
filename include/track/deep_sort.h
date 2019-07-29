#ifndef _DEEP_SORT_H_
#define _DEEP_SORT_H_
#include <iostream>
#include <set>
#include <vector>
#include "byavs.h"
#include "kalman_tracker.h"
#include "hungarian.h"

typedef struct{
    int label;
    float score;
    cv::Rect_<float> bbox;
} DetObj;

class DeepSort{

    private:
        std::vector<KalmanTracker> m_trackers;
        std::vector<DetObj> m_detect;
        
        std::vector<std::vector<double>> m_cost_matrix;
        
        std::vector<cv::Point> m_match_pairs;
        std::vector<int> m_assign;

        std::set<int> m_all;
        std::set<int> m_matched;
        std::set<int> m_unmatched_det;
        std::set<int> m_unmatched_trk;

        int m_max_miss_time;

    public:

        void predict(cv::Mat);
        void computeDistance();
        void assignMatrix();
        void matchResult();
        void update(cv::Mat);
        void sendResult(cv::Mat, vector<TrackeKeyObject>&);

        double getIou(cv::Rect_<float>, cv::Rect_<float>);
        bool init(const std::string& model_dir, const TrackeParas& pas,
                        const int gpu_id);
        bool inference(const cv::Mat, const DetectObjects&,
                        std::vector<TrackeKeyObject>&);

}

#endif
