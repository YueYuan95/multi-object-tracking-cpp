#ifndef _CALIBRATOR_H_
#define _CALIBRATOR_H_

#include <NvInfer.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <string.h>
#include <iterator>
#include <fstream>

#include "Common.h"
#include "BatchStream.h"

using namespace nvinfer1;
namespace bdavs {
class Int8EntropyCalibrator : public IInt8EntropyCalibrator
{
public:
    Int8EntropyCalibrator(BatchStream& stream, int firstBatch, bool readCache = true);

    virtual ~Int8EntropyCalibrator();

    int getBatchSize() const override;

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override;

    const void* readCalibrationCache(size_t& length) override;

    void writeCalibrationCache(const void* cache, size_t length) override;

private:
    BatchStream mStream;
    bool mReadCache{true};

    size_t mInputCount;
    void* mDeviceInput{nullptr};

    std::vector<char> mCalibrationCache;
};
}
#endif //_CALIBRATOR_H_
