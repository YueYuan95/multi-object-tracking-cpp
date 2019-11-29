#ifndef _CUDA_FUN_PARAM_H_
#define _CUDA_FUN_PARAM_H_

/**  Device code**/
typedef struct {
    int id;
    int width;
    int height;
    int update_row; 
    float* elements;
} FeatureMatrix;

// Thread block size
#define BLOCK_SIZE 2

void EuclideanMetric(int n, const FeatureMatrix d_A, int idx_size, int *indexs,  const FeatureMatrix d_B, FeatureMatrix d_C, FeatureMatrix d_D);
void UpdateFeature(FeatureMatrix d_E, int idx, FeatureMatrix d_F);
void UpdateFeature(FeatureMatrix d_S, float * d_f);
void GetObjectFeature(int, FeatureMatrix, float *);
#endif
