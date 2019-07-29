#ifndef _DESCRIPTOR_H_
#define _DESCRIPTOR_H_
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "byavs.h"
#include "kalman_tracker.h"

class Descriptor{

    private:

        PedestrianFeature ped_feature;
        VehicleFeature veh_feature; 

    public:
        Descriptor(PedFeatureParas, std::string, VehFeatureParas, std::string, int);
        int updateFeature(cv::Mat, std::vector<KalmanTracker>&);

}

#endif