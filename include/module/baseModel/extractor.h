
#ifndef _EXTRACTOR_H_
#define _EXTRACTOR_H_

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "NetOperator.h"
namespace bdavs {
typedef struct {
    std::string name;
    std::vector<float> data;
} BlobData;

class Extractor : public NetOperator
{
public:
    bool handleResult(std::vector<Layer> &outputLayers,std::vector<std::vector<BlobData>>& result_blob_data_list,int image_number);

    bool inference(const std::vector<cv::Mat>& imgBGRs, std::vector<std::vector<BlobData> >& feats);
    bool inference(const cv::Mat &image,std::vector<float> &feat);
    bool inference(const std::vector<AVSGPUMat>& imgBGRAs, std::vector<std::vector<BlobData> >& feats);
};
}
#endif //_EXTRACTOR_H_
