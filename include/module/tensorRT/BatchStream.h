#ifndef _BATCH_STREAM_H_
#define _BATCH_STREAM_H_

#include <NvInfer.h>
#include <vector>
#include <algorithm>
#include <assert.h>
namespace bdavs {
class BatchStream
{
public:
    BatchStream(const std::string batchDir, int batchSize, int maxBatches);
    ~BatchStream();

    void reset(int firstBatch);
    bool next();
    void skip(int skipCount);

    float* getBatch();
    int getBatchesRead() const;
    int getBatchSize() const;

    nvinfer1::DimsNCHW getDims() const;

private:
    float* getFileBatch();

    bool update();

private:
    int mBatchSize{0};
    int mMaxBatches{0};
    int mBatchCount{0};

    int mFileCount{0};
    int mFileBatchPos{0};

    int mImageSize{0};

    nvinfer1::DimsNCHW mDims;

    std::string mBatchDir;

    std::vector<float> mBatch;
    std::vector<float> mFileBatch;
};
}
#endif // _BATCH_STREAM_H_
