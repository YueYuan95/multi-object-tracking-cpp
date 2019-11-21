#ifndef __SORT_TRACKER_H__
#define __SORT_TRACKER_H__

#include <iostream>
#include <vector>
#include <math.h>
#include <set>
#include <algorithm>

#include "sort_base_tracker.h"
#include "kalman_tracker.h"
#include "hungarian.h"
#include "feature_extract.h"
#include "util.h"
#include "byavs.h"

using namespace sort;

class SortTracker{

    public:
        int init(byavs::PedFeatureParas, std::string, int);

        int inference(const byavs::TrackeInputGPU, byavs::TrackeObjectGPUs&);

        friend int crop_gpu_mat(byavs::GpuMat, std::vector<cv::Rect_<float>>, 
                       std::vector<bdavs::AVSGPUMat>&);
                       
        friend double get_feature_distance(std::vector<float> feature_a, std::vector<float> feature_b);
    
    private:
        
        std::vector<Tracker> m_trackers;

        std::vector<int> m_tracked_trackers;
        std::vector<int> m_lost_trackers;
        std::vector<int> m_removed_trackers;

        FeatureExtract extractor;

        int generate_candidate_trackers(std::vector<int>&, std::vector<int>, std::vector<int>);
        int compute_detection_feature(byavs::GpuMat, std::vector<cv::Rect_<float>>, 
                                      std::vector<std::vector<float>>&);
        int compute_feature_distance(std::vector<std::vector<double>>&, std::vector<int>, std::vector<cv::Rect_<float>>,
                                     std::vector<std::vector<float>>);
        int compute_iou_distance(std::vector<std::vector<double>>&, std::vector<int>, 
                                 std::vector<cv::Rect_<float>>);
        int matching(std::vector<std::vector<double>>, std::map<int, int>&, std::vector<int>&, 
                    std::vector<int>&);

        int deal_duplicate_tracker(std::vector<int>&, std::vector<int>);
};

#endif
