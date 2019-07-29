#include"byavs.h"
#include"src/module/tensorRT/extractor.h"
bool VehicleFeature::init(const std::string& model_dir,const VehFeatureParas & pas,const int gpu_id)
{
	Extractor* extractor;
	NetParameters para;
    para.input_n = 1;
    para.input_c = 3;
    para.input_h = 320;
    para.input_w = 320;
    para.name = "vehicle_reid";
    para.mean_val = {
        123.675,116.28,103.53
    };
    para.scale = 1.0/255.0f;
    para.input_layer_name = "blob1";
    para.output_layer_names =
            { "batch_norm_blob54"};
    float3 scales= {58.395,57.12,57.375};
    extractor = new Extractor();
    extractor->init(para,scales);

    std::string cacheFile = model_dir+"/vehicle_reid/VehicleReID_model_new.engine";
    std::cout<<cacheFile<<std::endl;
	if (!extractor->modelCacheExists(cacheFile.c_str()))
	{
        std::cout<<cacheFile<<" not exists"<<std::endl;
        return false;
	}
    extractor->loadModel(cacheFile.c_str(), gpu_id);
	extractor_model = (void*)extractor;
    return true;
}
bool VehicleFeature::inference(const ImgBGRArray& imgBGRs, float** vehFeatures)
{
	Extractor* extractor=(Extractor*)extractor_model;
    int num=0;
	for (int i=0;i<imgBGRs.size();i++)
    {
        
        std::vector<float> features;
        extractor->inference(imgBGRs[i], features);
        for (int j=0;j<features.size();j++)
        {
            (*vehFeatures)[num+j]=features[j];
        }
        num=num+features.size();
        // for (int j=0;j<features.size();j++)
        // {
        // 	std::cout<<features[j]<<std::endl;
        // }
    }
    return true;
}
void VehicleFeature::release()
{
    if (extractor_model!=NULL)
    {
        Extractor* extractor=(Extractor*)extractor_model;
        delete extractor;
    }
    return;
}
