#include"byavs.h"
#include"src/module/tensorRT/extractor.h"
bool PedestrianFeature::init(const std::string& model_dir,const PedFeatureParas & pas,const int gpu_id)
{
	Extractor* extractor;
	NetParameters para;
    para.input_n = 1;
    para.input_c = 3;
    para.input_h = 256;
    para.input_w = 128;
    para.name = "pedestrian_reid";
    para.mean_val = {
        0.445*255,0.456*255,0.406*255
    };
    para.scale = 1.0/255.0f;
    para.input_layer_name = "blob1";
    para.output_layer_names =
            { "batch_norm_blob54"};
    extractor = new Extractor();
    float3 scales= {1.0/255.0/0.229,1.0/255.0/0.224,1.0/255.0/0.225};
    extractor->init(para,scales);

    std::string cacheFile = model_dir+"/person_reid/PersonReID_model_new.engine";
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
bool PedestrianFeature::inference(const ImgBGRArray& imgBGRs, float** pedFeatures)
{
    Extractor* extractor=(Extractor*)extractor_model;
    int num=0;
	for (int i=0;i<imgBGRs.size();i++)
    {
        
        std::vector<float> features;
        extractor->inference(imgBGRs[i], features);
        for (int j=0;j<features.size();j++)
        {
            (*pedFeatures)[num+j]=features[j];
        }
        num=num+features.size();
        // for (int j=0;j<features.size();j++)
        // {
        // 	std::cout<<features[j]<<std::endl;
        // }
    }
    return true;
}
void PedestrianFeature::release()
{
    if (extractor_model!=NULL)
    {
        Extractor* extractor=(Extractor*)extractor_model;
        delete extractor;
    }
    return;
}