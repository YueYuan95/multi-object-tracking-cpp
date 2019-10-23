#ifndef __SORT_TRACKER_H_
#define __SORT_TRACKER_H_

#include <iostream>
#include <vector>
#include <math.h>
#include <set>
#include <algorithm>

#include "kalman_tracker.h"
#include "hungarian.h"
#include "util.h"
#include "byavs.h"

#define USE_DEEP 0
#define MAX_MISS_TIME 30

class SORT_tracker{

    public:
        int inference(const std::string& model_dir, const byavs::TrackeParas& pas,                      const int gpu_id);
        int inference();
    
    private:
        
        std::vector<Tracker> m_trackers;

        std::vector<int> m_tracked_trackers;
        std::vector<int> m_lost_trackers;
        std::vector<int> m_removed_trackers;

        int generate_candidate_trackers(std::vector<int>&, std::vector<int>, std::vector<int>);
        int compute_detection_feature(cv::Mat, std::vector<cv::Rect_<float>>, 
                                      std::vector<std::vector<float>>&);
        int compute_feature_distance(std::vector<std::vector<double>>&, std::vector<int>, 
                                     std::vector<std::vector<float>>);
        int compute_iou_distance(std::vector<std::vector<double>>&, std::vector<int>, 
                                 std::vector<cv::Rect_<float>>);
        int matching(std::vector<std::vector<double>>, std::map<int, int>&, std::vector<int>&, 
                    std::vector<int>&);

        int deal_duplicate_tracker(std::vector<int>&, std::vector<int>);
};

#endif
