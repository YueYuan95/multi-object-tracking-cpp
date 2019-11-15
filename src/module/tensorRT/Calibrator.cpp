
#include "Calibrator.h"
namespace bdavs {
/**
 * @brief Int8EntropyCalibrator::Int8EntropyCalibrator
 * @param stream
 * @param firstBatch
 * @param readCache
 */
Int8EntropyCalibrator::Int8EntropyCalibrator(BatchStream& stream, int firstBatch, bool readCache)
    : mStream(stream),
      mReadCache(readCache)
{
    DimsNCHW dims = mStream.getDims();
    mInputCount = mStream.getBatchSize() * dims.c() * dims.h() * dims.w();

    CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));

    mStream.reset(firstBatch);
}

/**
 * @brief Int8EntropyCalibrator::~Int8EntropyCalibrator
 */
Int8EntropyCalibrator::~Int8EntropyCalibrator()
{
    CHECK(cudaFree(mDeviceInput));
}

/**
 * @brief Int8EntropyCalibrator::getBatchSize
 * @return
 */
int Int8EntropyCalibrator::getBatchSize() const
{
    return mStream.getBatchSize();
}

/**
 * @brief Int8EntropyCalibrator::getBatch
 * @param bindings
 * @param names
 * @param nbBindings
 * @return
 */
bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings)
{
    if (!mStream.next())
        return false;

    CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));

//    assert(!strcmp(names[0], "data"));
    bindings[0] = mDeviceInput;
    return true;
}

/**
 * @brief Int8EntropyCalibrator::readCalibrationCache
 * @param length
 * @return
 */
const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length)
{
    mCalibrationCache.clear();
    std::ifstream input("calibration.table", std::ios::binary);
    input >> std::noskipws;
    if (mReadCache && input.good())
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));

    length = mCalibrationCache.size();
    return length ? &mCalibrationCache[0] : nullptr;
}

/**
 * @brief Int8EntropyCalibrator::writeCalibrationCache
 * @param cache
 * @param length
 */
void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length)
{
    std::ofstream output("calibration.table", std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}
}