#ifndef _MHT_TRACKER_H_
#define _MHT_TRACKER_H_

#include <iostream>
#include <opencv2/core/core.hpp>
#include "mht_tree.h"

#include "byavs.h"

class MHT_tracker{

    public:
        
        int N = 3;

        bool init(const std::string& model_dir, const byavs::TrackeParas& pas, const int gpu_id);
        bool inference(const byavs::TrackeInputGPUArray& inputs, byavs::TrackeResultGPUArray& resultArray);
        bool inference(const byavs::TrackeInputCPUArray& inputs, byavs::TrackeResultCPUArray& resultArray);

        /*
        *   intput : TrackeInputGPU
        *   output : TrackeResultGPU
        */
        int inference();

        /*
         *  input :  detect result , vector<cv::Rect_<float>>
         *           track tree, vector<Tree>
         *  output : track tree, vector<Tree>
         * 
         */
        int construct();
        void gating(std::vector<cv::Rect_<float>> det_result, std::vector<Tree>& tree_list);
        /*
        * I think construct contain the gating and scoring 
        int gating(std::vector<cv::Rect_<float> det_result, std::vector<Tree> tree_list);
        int scoring();
        */
        int sovle_mwis();
        int pruning();
        int morls();
};

#endif
