
#ifndef _SELECTOR_H_
#define _SELECTOR_H_

#include "NetOperator.h"
namespace bdavs {

class Selector : public NetOperator {
public:
    bool inference(const AVSGPUMat& imgBGR, int &label) {
      
        if(!preprocess_gpu(imgBGR))
        {
            printf("[TensorNet] preprocess Image Failed\n");
            return false;
        }

        // inference
        context->execute(mDims.n(), mBuffers);
        float maxScore = 0.f;
        int maxIndex = -1;
        for(size_t j=0; j < mOutputLayers[0].dims.c(); j++)
        {
            float *data;
            data=mOutputLayers[0].top_data;
            if(data[j] > maxScore)
            {
                maxScore =data[j];
                maxIndex = j;
            }
        }

	 label = maxIndex;
	// printf("select:maxindex:%d,label:%d\n",maxIndex,label);
        if(maxIndex == 10)
            return false;

        return true;
    }
};
}
#endif //_SELECTOR_H_
