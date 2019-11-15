#ifndef _FEATRUE_EXTRACTOR_H_
#define _FEATRUE_EXTRACTOR_H_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "NetOperator.h"
#include "byavs.h"
#include "extractor.h"
#include "kalman_tracker.h"
#include "util.h"

class FeatureExtract{
    
    private:

        bdavs::Extractor * extractor;

    public:

        int init(byavs::PedFeatureParas, std::string, int);
        bool inference(std::vector<bdavs::AVSGPUMat>, std::vector<std::vector<float>>&);
        int release();
};

#endif
