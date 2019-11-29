#ifndef _MULTI_TRACKER_GPU_H_
#define _MULTI_TRACKER_GPU_H_

#include<iostream>
#include<vector>
#include<set>

#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "byavs.h"
#include "kalman_tracker_v2.h"
#include "base_tracker_gpu.h"
#include "nn_matching_gpu.h"
#include "linear_assignment_gpu.h"
#include "feature_extract.h"
#include "tracker_param.h"
#include "cuda_fun_param.h"
#include "util.h"

class MultiTrackerGPU{

    private:
        KalmanTrackerV2 kf;
        FeatureExtract extractor;
        DistanceMetricGPU dm;
        LinearAssignmentGPU la;
        std::vector<BaseTrackerGPU> m_tracker_list;
        int next_id = 1;
        int cascade_depth = 30;
        float *m_single_feature;
        FeatureMatrix detection_feature;
    
    public:
        int inference(const byavs::TrackeInputGPU, byavs::TrackeObjectGPUs&);

        int init(byavs::PedFeatureParas, std::string, int);
        int match(std::map<int,int>& matches, std::vector<int>& um_trackers, std::vector<int>& um_detection,
                std::vector<cv::Rect_<float>> detection_boxes, FeatureMatrix detection_feature);
        int initiate_tracker(int label, cv::Rect_<float> detection_box, float* detection_feature);
        int compute_detection_feature(byavs::GpuMat, std::vector<cv::Rect_<float>>, 
                FeatureMatrix&);
        int get_object_feature(int, FeatureMatrix, float *);
        int release();

};

#endif
