//
// Created by xing on 19-5-1.
//
#include <algorithm>
#include "fssd.h"
namespace bdavs {
bool cmpBBox(const DObject b1, const DObject b2)
{
    return (b1.score>b2.score);
}
int nonMaxRectSuppression(std::vector<DObject> &bboxes, double threshold, char type)
{
    std::vector<DObject> bboxes_nms;
    std::sort(bboxes.begin(), bboxes.end(), cmpBBox);

    int32_t select_idx = 0;
    int32_t num_bbox = static_cast<int32_t>(bboxes.size());
    std::vector<int32_t> mask_merged(num_bbox, 0);
    bool all_merged = false;

    while (!all_merged) {
        while (select_idx < num_bbox && mask_merged[select_idx] == 1)
            select_idx++;
        if (select_idx == num_bbox) {
            all_merged = true;
            continue;
        }

        bboxes_nms.push_back(bboxes[select_idx]);
        mask_merged[select_idx] = 1;

        DObject select_bbox = bboxes[select_idx];
        float area1 = static_cast<float>((select_bbox.box.x2-select_bbox.box.x1+1) * (select_bbox.box.y2-select_bbox.box.y1+1));
        float x1 = static_cast<float>(select_bbox.box.x1);
        float y1 = static_cast<float>(select_bbox.box.y1);
        float x2 = static_cast<float>(select_bbox.box.x2);
        float y2 = static_cast<float>(select_bbox.box.y2);

        select_idx++;
        for (int32_t i = select_idx; i < num_bbox; i++) {
            if (mask_merged[i] == 1)
                continue;

            DObject& bbox_i = bboxes[i];
            float x = std::max<float>(x1, static_cast<float>(bbox_i.box.x1));
            float y = std::max<float>(y1, static_cast<float>(bbox_i.box.y1));
            float w = std::min<float>(x2, static_cast<float>(bbox_i.box.x2)) - x + 1;
            float h = std::min<float>(y2, static_cast<float>(bbox_i.box.y2)) - y + 1;
            if (w <= 0 || h <= 0)
                continue;

            float area2 = static_cast<float>((bbox_i.box.x2-bbox_i.box.x1+1) * (bbox_i.box.y2-bbox_i.box.y1+1));
            float area_intersect = w * h;

            switch (type) {
                case 'u':
                    if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > threshold)
                        mask_merged[i] = 1;
                    break;
                case 'm':
                    if (static_cast<float>(area_intersect) / std::min(area1 , area2) > threshold)
                        mask_merged[i] = 1;
                    break;
                default:
                    break;
            }
        }
    }

    bboxes = bboxes_nms;

    return 0;
}
bool FSSD::parseOutput(std::vector<Layer> &outputLyaers,std::vector<std::vector<DObject>>& detectResults,std::vector<cv::Size> image_size_list,float confidence, int minW, int minH)
{
    for(size_t n=0; n<image_size_list.size(); n++)
    {
        std::vector<DObject> detectResult;
        detectResult.clear();

        float* output = mOutputLayers[0].top_data + n*mOutputLayers[0].dims.c()*mOutputLayers[0].dims.h()*mOutputLayers[0].dims.w();

        for(int h=0; h < mOutputLayers[0].dims.h(); h++)
        {
            if(*(output + 2 + h * 7) >= confidence)
            {
#ifdef DEBUG
//                printf("[TensorNet] Detection output %d [id = %f , label = %f , score = %f , xMin = %f , yMin = %f , xMax = %f , yMax = %f\n",
//                       h,
//                       *(output + h * 7),
//                       *(output + 1 + h * 7),
//                       *(output + 2 + h * 7),
//                       *(output + 3 + h * 7),
//                       *(output + 4 + h * 7),
//                       *(output + 5 + h * 7),
//                       *(output + 6 + h * 7));
#endif
                DObject dObject;

                dObject.label = *(output + 1 + h * 7);
                dObject.score = *(output + 2 + h * 7);
                dObject.box.x1 = int(*(output + 3 + h * 7) * image_size_list[n].width);
                dObject.box.y1 = int(*(output + 4 + h * 7) * image_size_list[n].height);
                dObject.box.x2 = int(*(output + 5 + h * 7) * image_size_list[n].width);
                dObject.box.y2 = int(*(output + 6 + h * 7) * image_size_list[n].height);

                if((dObject.box.x2 - dObject.box.x1) <= minW || (dObject.box.y2 - dObject.box.y1) <= minH)
                    continue;

                detectResult.push_back(dObject);
            }
        }
        nonMaxRectSuppression(detectResult,0.8,'u');
        detectResults.push_back(detectResult);
    }
    return true;
}

bool FSSD::inference(const std::vector<cv::Mat>& imgBGRs, std::vector<std::vector<DObject>>& detectResults,
                         float confidence, int minW, int minH)
{
    // printf("fssd inference start");
    std::vector<std::vector<cv::Mat> > image_list;
    getHandleImages(imgBGRs,image_list);
    // preprocess
    for (int k=0;k<image_list.size();k++)
    {
        if(!preprocess_gpu(image_list[k])) {
        printf("[TensorNet] preprocess Image Failed\n");
        return false;
        }
        context->execute(image_list[k].size(), mBuffers);
        std::vector<cv::Size> image_size_list;
        for (int i=0;i<image_list[k].size();i++)
        {
            image_size_list.push_back(cv::Size(image_list[k][i].cols,image_list[k][i].rows));
        }
        // parse output
        parseOutput(mOutputLayers,detectResults,image_size_list,confidence,minW,minH);
    }
    std::cout<<detectResults.size()<<" "<<imgBGRs.size() <<std::endl;
    assert(detectResults.size()==imgBGRs.size());
    

    return true;
}

bool FSSD::inference(const std::vector<AVSGPUMat>& imgBGRAs, std::vector<std::vector<DObject>>& detectResults,
                         float confidence, int minW, int minH)
{
    std::vector<std::vector<AVSGPUMat> > image_list;
    getHandleImages(imgBGRAs,image_list);
    for (int k=0;k<image_list.size();k++){
        if(!preprocess_gpu(image_list[k]))
        {
            printf("[TensorNet] preprocess Image Failed\n");
            return false;
        }

        context->execute(image_list[k].size(), mBuffers);

        std::vector<cv::Size> image_size_list;
        for (int i=0;i<image_list[k].size();i++)
        {
            image_size_list.push_back(cv::Size(image_list[k][i].width,image_list[k][i].height));
        }
        // parse output
        parseOutput(mOutputLayers,detectResults,image_size_list,confidence,minW,minH);
    }
    // std::cout<<detectResults.size()<<" "<<imgBGRAs.size() <<std::endl;
     assert(detectResults.size()==imgBGRAs.size());
    return true;
    }
}
