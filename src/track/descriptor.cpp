#include "descriptor.h"

Descriptor::Descriptor(PedFeatureParas ped_feature_paras, std::string ped_model_dir, 
                    VehFeatureParas veh_feature_paras, std::string veh_model_dir, int gpuId){

    ped_feature.init(model_dir, ped_feature_paras, gpuId);
    veh_feature.init(model_dir, veh_feature_paras, gpuId);

}

int Descriptor::updateFeature(cv::Mat image, std::vector<KalmanTracker>& tracker_list){

    std::vector<int> ped_index;
    std::vector<int> veh_index;

    ImgBGRArray ped_images;
    ImgBGRArray veh_images;

    /* 
        figure out pedestrain and car
        and only update time since update is 0
    */
    for(int i=0; i < tracker_list.size(); i++){
        if(tracker_list[i].getTime() == 0){
            if(tracker_list[i].label == 2){
                ped_index.insert(i);
            }
            if(tracker_list[i].label == 0){
                veh_index.insert(i);
            }
        }
    }

    /*cut ROI */
    for(int i=0; i < ped_index.size(); i++){
        cv::Mat roi = image(tracker_list[ped_index[i]].getBbox());
        ped_images.push_back(roi.copy());
    }

    for(int i=0; i < veh_index.size(); i++){
        cv::Mat roi = image(tracker_list[veh_index[i]].getBBox());
        veh_images.push_back(roi.copy());
    }

    /*malloc memory */
    float** ped_features = (float**)malloc(ped_index.size()*sizeof(float*));
    for(int i = 0; i < ped_index.size(); i++){
        ped_features[i] = (float*)malloc(2048*sizeof(float*));
    }

    float** veh_features = (float**)malloc(veh_index.size()*sizeof(float*));
    for(int i = 0; i < veh_index.size(); i++){
        veh_features[i] = (float*)malloc(2048*sizeof(float*));
    }

    /*inference and get feature */
    ped_feature.inference(ped_images, ped_features);
    veh_feature.inference(veh_images, veh_features);

    /*assign descriptor*/
    for(int i=0; i < ped_index.size(); i++){
        std::shared_ptr<float> p(ped_features[i]);
        tracker_list[ped_index[i]].setDescriptor(p);     
    }
    for(int i=0; i < veh_index.size(); i++){
        std::shared_ptr<float> v(veh_features[i]);
        tracker_list[veh_index[i]].setDescriptor(v);
    }

    /*release */
    // free ped_features;
    // free veh_features;

    ped_feature.release();
    veh_feature.release();

}