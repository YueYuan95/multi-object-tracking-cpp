#ifndef _DISTANCE_METRIC_GPU_H_
#define _DISTANCE_METRIC_GPU_H_

#include <iostream>
#include <map>

#include "base_tracker_gpu.h"
#include "tracker_param.h"
#include "cuda_fun_param.h"
#include "util.h"

class DistanceMetricGPU{

    private:

        int m_budget = 30;
        int m_max_detection = 50;
        int m_max_tracker = 50;
        int *m_detection_idxs, *m_d_detection_idxs;
        std::map<int, FeatureMatrix> m_samples;

        FeatureMatrix m_all_cost, m_d_single_cost, m_d_all_cost;


    public:
        DistanceMetricGPU();
        int partial_fit(int, int, FeatureMatrix);
        int partial_fit(int, float*);
        int distance(std::vector<std::vector<double>>&, std::vector<int>,
                    FeatureMatrix, std::vector<int>);
        int distance(std::vector<std::vector<double>>&, std::vector<cv::Rect_<float>>, 
                    std::vector<BaseTrackerGPU>);

        int remove_object(int);
        int release();

};

#endif