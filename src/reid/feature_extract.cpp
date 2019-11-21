#include "feature_extract.h"

int FeatureExtract::init(byavs::PedFeatureParas ped_feature_paras, std::string ped_model_dir, 
                        int gpu_id){

    std::string config_path = "caffe2tensorRT.param";
    std::string ped_engine_dir=ped_model_dir+"/PersonReID_test";             // /PersonReID_test
    extractor = new bdavs::Extractor();
    extractor->readConfig(ped_engine_dir,config_path);
    if (extractor->loadModel(gpu_id)==false)
    {
        std::cout<<"Ped Extractor Load Failed!"<<std::endl;   
    }

}

bool FeatureExtract::inference(std::vector<bdavs::AVSGPUMat> gpu_mats, std::vector<std::vector<float>>& feature_list){

    std::vector<std::vector<bdavs::BlobData> > feats_result;

    extractor->inference(gpu_mats,feats_result);
    for(int i=0;i<feats_result.size();i++) {
        assert(feats_result[i].size()==1);
        for(int j=0; j<feats_result[i].size();j++){
            //std::vector<float> feature = norm(feats_result[i][j].data);
            feature_list.push_back(feats_result[i][j].data);
        }
    }

}

int FeatureExtract::release(){

    if(extractor != NULL) delete extractor;
    return 1;
}