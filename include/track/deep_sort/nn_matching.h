#ifndef _DISTANCE_METRIC_H_
#define _DISTANCE_METRIC_H_

#include<iostream>
#include<map>
#include<vector>
#include<assert.h>
#include<algorithm>

#include "base_tracker.h"
#include "util.h"

class DistanceMetric{

    private:

        int m_budget = 30;
        std::map<int, std::vector<std::vector<float>>> m_samples;

    public:

        int partial_fit(std::vector<std::vector<float>>, std::vector<int>);
        int distance(std::vector<std::vector<double>>&, std::vector<std::vector<float>>, 
                    std::vector<int>);
        int distance(std::vector<std::vector<double>>&, std::vector<cv::Rect_<float>>, 
                    std::vector<BaseTracker>);

        double compute_cosine_distance(std::vector<float>, std::vector<float>);
        double compute_euclidean_distance(std::vector<float>, std::vector<float>);

};

#endif