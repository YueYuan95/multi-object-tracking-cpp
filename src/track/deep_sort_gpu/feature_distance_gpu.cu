#include "cuda_fun_param.h"

// Forward declaration of the matrix multiplication kernel
__global__ void EuclideanMetricKernel(const FeatureMatrix, int, int*, const FeatureMatrix,  FeatureMatrix);
__global__ void FindMixNumberKernel(int, const FeatureMatrix, FeatureMatrix);
__global__ void UpdateFeatureKernel(int, FeatureMatrix, int, FeatureMatrix);
__global__ void GetObjectFeatureKernel(int, FeatureMatrix, float*);

// Matrix multiplication kernel called by MatMul()
__global__ void EuclideanMetricKernel(FeatureMatrix A, int idx_size, int *indexs, FeatureMatrix B, FeatureMatrix C)
{
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int i;
    for(i=0; i < idx_size; i++){
        if(col == indexs[i]) break;
    }
    if(i == idx_size) return;

    if (row>=A.height||col>=B.width || row >= C.height || col > C.width) return;

    for (int e = 0; e < A.width; ++e){
        Cvalue += (A.elements[row * A.width + e] - B.elements[e * B.width + col])
                * (A.elements[row * A.width + e] - B.elements[e * B.width + col]);
        // Cvalue += (A.elements[row * A.width + e] * B.elements[e * B.width + col]);
    }

    C.elements[row * C.width + i] = Cvalue;

    /************Method 2***********/

    // float Cvalue = 0;
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // int col;
    // for(int i=0; i < idx_size; i++){
    //     col = indexs[i];

    //     if (row>=A.height||col>=B.width || row >= C.height || col > C.width) return;

    //     for (int e = 0; e < A.width; ++e){
    //         Cvalue += (A.elements[row * A.width + e] - B.elements[e * B.width + col])
    //             * (A.elements[row * A.width + e] - B.elements[e * B.width + col]);
    //          // Cvalue += (A.elements[row * A.width + e] * B.elements[e * B.width + col]);
    //     }

    //     C.elements[row * C.width + i] = Cvalue;
    // }

}

// Find mixmum number of every cols of Matrix
// Device code
__global__ void FindMixNumberKernel(int n, const FeatureMatrix C, FeatureMatrix D){

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col < C.width){
        float min_value = 1e5;
        for(int e = 0; e < C.height; e++){
            if(min_value > C.elements[e*C.width + col]){
                min_value = C.elements[e*C.width + col];
            }
        }
        D.elements[n * D.width + col] = min_value;
    }
}

// Update the feature of tracking object
// Device code
__global__ void UpdateFeatureKernel(int h, FeatureMatrix E, int idx, FeatureMatrix F){

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col < E.width && col < F.height){
        E.elements[h*E.width + col] = F.elements[col * F.width + idx];
    }
}

__global__ void UpdateFeatureKernel_V2(int h, FeatureMatrix E, float *update_feature){

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col < E.width){
        E.elements[h*E.width + col] = update_feature[col];
    }
}


__global__ void GetObjectFeatureKernel(int index, FeatureMatrix D_F, float *d_f){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < D_F.height){
        d_f[row] = D_F.elements[row*D_F.width + index];
    }

}

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void EuclideanMetric(int n, const FeatureMatrix d_A, int idx_size, int *indexs, const FeatureMatrix d_B, FeatureMatrix d_C, FeatureMatrix d_D)
{
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((d_B.width+BLOCK_SIZE-1) / BLOCK_SIZE , (d_A.height+BLOCK_SIZE-1) / BLOCK_SIZE );

    EuclideanMetricKernel<<<dimGrid, dimBlock>>>(d_A, idx_size, indexs, d_B, d_C);
    //Not sure if don't use cudaDeviceSynchronize(), the result is right
    //cudaDeviceSynchronize();
    FindMixNumberKernel<<<dimGrid,dimBlock>>>(n, d_C, d_D);
}

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void UpdateFeature(FeatureMatrix d_E, int idx, FeatureMatrix d_F){

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((d_E.width+BLOCK_SIZE-1) / BLOCK_SIZE , (d_E.width+BLOCK_SIZE-1) / BLOCK_SIZE );

    UpdateFeatureKernel<<<dimGrid,dimBlock>>>(d_E.update_row, d_E, idx, d_F);

}

void UpdateFeature(FeatureMatrix d_S, float * d_f){

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((d_S.width+BLOCK_SIZE-1) / BLOCK_SIZE , (d_S.width+BLOCK_SIZE-1) / BLOCK_SIZE );

    UpdateFeatureKernel_V2<<<dimGrid,dimBlock>>>(d_S.update_row, d_S, d_f);

}

void GetObjectFeature(int index, FeatureMatrix D_F, float * o_f){

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((D_F.height+BLOCK_SIZE-1)/BLOCK_SIZE, (D_F.height+BLOCK_SIZE-1)/BLOCK_SIZE);
    GetObjectFeatureKernel<<<dimGrid, dimBlock>>>(index, D_F, o_f);
}