#ifndef _MHT_TRACKER_H_
#define _MHT_TRACKER_H_

#include <iostream>
#include <math.h>
#include <opencv2/core/core.hpp>
#include "mht_tree.h"
#include "mht_graph.h"
#include "util.h"
#include "byavs.h"
#include "kalman_tracker.h"
#include "hungarian.h"
class MHT_tracker{

    private:

        std::vector<Tree> tree_list;
        int N = 10;
        int miss_time_thrd = N+10;////
        int hit_time_thrd =  N-(N/2);////


    public:
        
        
        bool init(const std::string& model_dir, const byavs::TrackeParas& pas, const int gpu_id);
        bool inference(const byavs::TrackeInputGPUArray& inputs, byavs::TrackeResultGPUArray& resultArray);
        bool inference(const byavs::TrackeInputCPUArray& inputs, byavs::TrackeResultCPUArray& resultArray);
        

        /*
        *   intput : TrackeInputGPU
        *   output : TrackeResultGPU
        */
        //int inference(std::vector<cv::Rect_<float>>, byavs::TrackeObjectCPUs&);
        ///int inference(std::vector<cv::Rect_<float>> det_result, std::vector<float> det_result_score, byavs::TrackeObjectCPUs& results);
        int inference(std::vector<cv::Rect_<float>>, std::vector<float>,byavs::TrackeObjectCPUs& , byavs::TrackeObjectCPUs& );

        /*
         *  input :  detect result , vector<cv::Rect_<float>>
         *           track tree, vector<Tree>
         *  output : track tree, vector<Tree>
         * 
         */
        int construct();
        std::vector<cv::Rect_<float>> NMS(std::vector<cv::Rect_<float>>, std::vector<float>);
        ///int gating(std::vector<cv::Rect_<float>>);
        int gating(std::vector<cv::Rect_<float>> , byavs::TrackeObjectCPUs& );

        int backTraversal(treeNode, std::shared_ptr<treeNode>, std::vector<int>&, std::vector<float>&, std::vector<std::vector<int>>&, std::vector<std::vector<float>>&, int);
        int TreeToGraph(Graph&);
        /*
        * I think construct contain the gating and scoring 
        int gating(std::vector<cv::Rect_<float> det_result, std::vector<Tree> tree_list);
        int scoring();
        */
        int sovle_mwis(Graph, std::map<int, std::vector<int>>&);
        int pruning(std::map<int, std::vector<int>>);
        int morls();

        int sentResult(byavs::TrackeObjectCPUs&);
        //std::vector<std::vector<double>> computeDistance(std::vector<std::shared_ptr<treeNode>>, std::vector<cv::Rect_<float>>); 
        std::vector<std::vector<double>> computeDistance(std::vector<cv::Rect_<float>> , std::vector<cv::Rect_<float>> );

        //get function
        std::vector<Tree> get_tree_list();
};

#endif
