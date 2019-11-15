//
// Created by xing on 19-5-1.
//

#ifndef _FSSD_H_
#define _FSSD_H_

#include "NetOperator.h"

//#define DEBUG
namespace bdavs {
typedef struct
{
    int x1;
    int x2;
    int y1;
    int y2;
} Box;

typedef struct
{
    int label{-1};
    float score{0};
    Box box;
}DObject;

class FSSD : public NetOperator
{
public:
    bool parseOutput(std::vector<Layer> &outputLyaers,std::vector<std::vector<DObject> >& detectResults,std::vector<cv::Size> image_size_list,float confidence, int minW, int minH);
    bool inference(const std::vector<cv::Mat>& imgBGRs, std::vector<std::vector<DObject>>& detectResults,
                   float confidence, int minW, int minH);

    bool inference(const std::vector<AVSGPUMat>& imgBGRAs, std::vector<std::vector<DObject>>& detectResults,
                   float confidence, int minW, int minH);

};
}
#endif //_FSSD_H_
