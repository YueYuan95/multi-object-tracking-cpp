#ifndef _LINEAR_ASSIGNMENT_GPU_H_
#define _LINEAR_ASSIGNMENT_GPU_H_

#include<iostream>

#include "hungarian.h"
#include "nn_matching_gpu.h"
#include "base_tracker_gpu.h"
#include "cuda_fun_param.h"

class LinearAssignmentGPU{

    private:
        float max_feature_distance = 1.0;
        float max_iou_distance = 0.7;
        float gating_threshold = 9.4877;
        double gating_cost = 1e+5;
        HungarianAlgorithm hung_algo;

    public:
        int min_cost_matching(DistanceMetricGPU, KalmanTrackerV2, std::vector<cv::Rect_<float>>,
            std::map<int, int>&, std::vector<int>&, std::vector<int>&,
            FeatureMatrix, std::vector<int>,
            std::vector<BaseTrackerGPU>, std::vector<int>);
        int matching_cascade(DistanceMetricGPU dm, KalmanTrackerV2 kf,
            int cascade_depth, std::vector<cv::Rect_<float>> detection_boxes,
            std::map<int, int>& matches, std::vector<int>& um_detections, std::vector<int>& um_trackers,
            std::vector<BaseTrackerGPU> tracker_list, FeatureMatrix detection_feature,
            std::vector<int> tracker_index, std::vector<int> detection_index);
        int gate_cost_matrix(KalmanTrackerV2, std::vector<std::vector<double>>&,
            std::vector<BaseTrackerGPU>, std::vector<int>,
            std::vector<cv::Rect_<float>>, std::vector<int>);
        int matching_by_iou(DistanceMetricGPU dm,
            std::map<int, int>& matches, std::vector<int>& um_trackers, std::vector<int>& um_detections,
            std::vector<cv::Rect_<float>> detect_boxes, std::vector<int> detect_indexs,
            std::vector<BaseTrackerGPU> tracker_list, std::vector<int> tracker_indexs);
};

#endif