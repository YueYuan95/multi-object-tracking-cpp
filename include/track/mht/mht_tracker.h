#ifndef _MHT_TRACKER_H_
#define _MHT_TRACKER_H_

#include <iostream>
#include <opencv2/core/core.hpp>
#include "mht_tree.h"
#include "mht_graph.h"
#include "util.h"
#include "byavs.h"

class MHT_tracker{

    private:

        std::vector<Tree> tree_list;
        int N = 10;


    public:
        
        
        bool init(const std::string& model_dir, const byavs::TrackeParas& pas, const int gpu_id);
        bool inference(const byavs::TrackeInputGPUArray& inputs, byavs::TrackeResultGPUArray& resultArray);
        bool inference(const byavs::TrackeInputCPUArray& inputs, byavs::TrackeResultCPUArray& resultArray);

        /*
        *   intput : TrackeInputGPU
        *   output : TrackeResultGPU
        */
        int inference(std::vector<cv::Rect_<float>>, byavs::TrackeObjectCPUs&);

        /*
         *  input :  detect result , vector<cv::Rect_<float>>
         *           track tree, vector<Tree>
         *  output : track tree, vector<Tree>
         * 
         */
        int construct();
        int gating(std::vector<cv::Rect_<float>>);

        int backTraversal(treeNode, std::shared_ptr<treeNode>, std::vector<int>&,
                std::vector<std::vector<int>>&, int);
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

        //get function
        std::vector<Tree> get_tree_list();
};

#endif
