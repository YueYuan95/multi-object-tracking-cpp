#ifndef _MULTI_TRACKER_H_
#define _MULTI_TRACKER_H_

#include<iostream>
#include<vector>
#include<set>

#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "byavs.h"
#include "kalman_tracker_v2.h"
#include "base_tracker.h"
#include "nn_matching.h"
#include "linear_assignment.h"
#include "feature_extract.h"

class MultiTracker{

    private:
        KalmanTrackerV2 kf;
        FeatureExtract extractor;
        DistanceMetric dm;
        LinearAssignment la;
        std::vector<BaseTracker> m_tracker_list;
        int next_id = 1;
        int cascade_depth = 30;
    
    public:
        int inference(const byavs::TrackeInputGPU, byavs::TrackeObjectGPUs&);

        int init(byavs::PedFeatureParas, std::string, int);
        int match(std::map<int,int>& matches, std::vector<int>& um_trackers, std::vector<int>& um_detection,
                std::vector<cv::Rect_<float>> detection_boxes, std::vector<std::vector<float>> detection_feature);
        int initiate_tracker(int label, cv::Rect_<float> detection_box, std::vector<float> detection_feature);
        int compute_detection_feature(byavs::GpuMat, std::vector<cv::Rect_<float>>, 
                std::vector<std::vector<float>>&);

};

#endif
