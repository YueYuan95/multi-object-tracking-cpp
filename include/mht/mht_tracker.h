#include <iostream>

class MHT_tracker{

    public:
        
        bool init(const std::string& model_dir, const TrackeParas& pas, const int gpu_id);
        bool inference(const TrackeInputGPUArray& inputs, TrackeResultGPUArray& resultArray);
        bool inference(const TrackeInputCPUArray& inputs, TrackeResultCPUArray& resultArray);

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
        /*
        * I think construct contain the gating and scoring 
        int gating();
        int scoring();
        */
        int sovle_mwis();
        int pruning();
        int morls();
};