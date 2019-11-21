#ifndef _LINEAR_ASSIGNMENT_H_
#define _LINEAR_ASSIGNMENT_H_

#include<iostream>

#include "hungarian.h"
#include "nn_matching.h"
#include "base_tracker.h"

class LinearAssignment{

    private:
        float max_feature_distance = 1.0;
        float max_iou_distance = 0.7;
        float gating_threshold = 9.4877;
        double gating_cost = 1e+5;
        HungarianAlgorithm hung_algo;

    public:
        int min_cost_matching(DistanceMetric, KalmanTrackerV2, std::vector<cv::Rect_<float>>,
            std::map<int, int>&, std::vector<int>&, std::vector<int>&,
            std::vector<std::vector<float>>, std::vector<int>,
            std::vector<BaseTracker>, std::vector<int>);
        int matching_cascade(DistanceMetric dm, KalmanTrackerV2 kf,
            int cascade_depth, std::vector<cv::Rect_<float>> detection_boxes,
            std::map<int, int>& matches, std::vector<int>& um_detections, std::vector<int>& um_trackers,
            std::vector<BaseTracker> tracker_list, std::vector<std::vector<float>> detection_feature,
            std::vector<int> tracker_index, std::vector<int> detection_index);
        int gate_cost_matrix(KalmanTrackerV2, std::vector<std::vector<double>>&,
            std::vector<BaseTracker>, std::vector<int>,
            std::vector<cv::Rect_<float>>, std::vector<int>);
        int matching_by_iou(DistanceMetric dm,
            std::map<int, int>& matches, std::vector<int>& um_trackers, std::vector<int>& um_detections,
            std::vector<cv::Rect_<float>> detect_boxes, std::vector<int> detect_indexs,
            std::vector<BaseTracker> tracker_list, std::vector<int> tracker_indexs);
};

#endif