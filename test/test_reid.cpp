#include "test.h"

int test_extract(){
    
    int gpu_id = 0;
    bdavs::Extractor* ped_extractor;

    std::string config_path = "caffe2tensorRT.param";
    std::string ped_model_dir = "/root/data";
    std::string ped_engine_dir=ped_model_dir+"/PersonReID_test";
    ped_extractor = new bdavs::Extractor();
    ped_extractor->readConfig(ped_engine_dir,config_path);
    if (ped_extractor->loadModel(gpu_id)==false)
    {
        std::cout<<"Ped Extractor Load Failed!"<<std::endl;   
    }

    std::vector<bdavs::AVSGPUMat> gpu_mats; 
    get_gpu_mats(gpu_mats);

    std::vector<std::vector<bdavs::BlobData> > feats_result;
    std::vector<std::vector<float>> feature_list;

    ped_extractor->inference(gpu_mats,feats_result);
    for(int i=0;i<feats_result.size();i++) {
        assert(feats_result[i].size()==1);
        for(int j=0; j<feats_result[i].size();j++){
            // std::cout<<feats_result[i][j].data.size()<<std::endl;
            // for(auto k : feats_result[i][j].data) std::cout<<k<<" ";
            // std::cout<<std::endl;
            feature_list.push_back(feats_result[i][j].data);
        }
    }

    std::vector<std::vector<double>> cost_matrix;
    get_feature_cost_matrix(cost_matrix, feature_list, feature_list);
    for(int i=0; i<cost_matrix.size();i++){
        for(int j=0; j<cost_matrix[i].size();j++){
            std::cout<<cost_matrix[i][j]<<" ";
        }
        std::cout<<std::endl;
    }

    //TODO: Free GPU Memory
}

int test_feature_extractor(){

    std::string model_dir = "/root/data";
    byavs::PedFeatureParas ped_feature_param;
    int gpu_id = 0;
    FeatureExtract extractor;
    extractor.init(ped_feature_param, model_dir, gpu_id);

    // byavs::GpuMat gpu_mat;
    // std::vector<bdavs::AVSGPUMat> gpu_mats;
    // std::vector<cv::Rect_<float>> object_boxes;
    // get_gpu_mat(gpu_mat);
    // get_bounding_boxes(object_boxes);
    // crop_gpu_mat(gpu_mat, object_boxes, gpu_mats);

    std::vector<bdavs::AVSGPUMat> gpu_mats; 
    get_gpu_mats(gpu_mats);

    std::vector<std::vector<float>> feature_list;
    extractor.inference(gpu_mats, feature_list);
    // for(int i=0;i<feature_list.size();i++) {
    //     std::cout<<feature_list[i].size()<<std::endl;
    //     for(auto k : feature_list[i]) std::cout<<k<<" ";
    //     std::cout<<std::endl;
    // }

    std::vector<std::vector<double>> cost_matrix;
    get_feature_cost_matrix(cost_matrix, feature_list, feature_list);
    for(int i=0; i<cost_matrix.size();i++){
        for(int j=0; j<cost_matrix[i].size();j++){
            std::cout<<cost_matrix[i][j]<<" ";
        }
        std::cout<<std::endl;
    }

    //TODO: Free GPU Memory
}