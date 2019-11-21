
#include <cuda_runtime_api.h>

#include "Common.h"
namespace bdavs {
__global__ void gpuPreImageScaleMean(unsigned char* input, int inputWidth, int inputHeight, int inputChannels,
                                   float* output, int outputWidth, int outputHeight,
                                   float3 scale, const float3 mean_value,const int color_type)
{
    //2D Index of current thread
    const int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if(dx < outputWidth && dy < outputHeight)
    {
        // BGRA or BGR
        if(inputChannels == 4 || inputChannels == 3 || inputChannels == 32 || inputChannels == 64)
        {

            double scale_x = (double) inputWidth / outputWidth;
            double scale_y = (double) inputHeight / outputHeight;

            int xmax = outputWidth;

            float fx = (float)((dx + 0.5) * scale_x - 0.5);
            int sx = floor(fx);
            fx = fx - sx;

            int isx1 = sx;
            if (isx1 < 0)
            {
                fx = 0.0;
                isx1 = 0;
            }
            if (isx1 >= (inputWidth - 1))
            {
                xmax = ::min( xmax, dy);
                fx = 0;
                isx1 = inputWidth - 1;
            }

            float2 cbufx;
            cbufx.x = (1.f - fx);
            cbufx.y = fx;

            float fy = (float)((dy + 0.5) * scale_y - 0.5);
            int sy = floor(fy);
            fy = fy - sy;

            int isy1 = clip(sy - 1 + 1 + 0, 0, inputHeight);
            int isy2 = clip(sy - 1 + 1 + 1, 0, inputHeight);

            float2 cbufy;
            cbufy.x = (1.f - fy);
            cbufy.y = fy;

            int isx2 = isx1 + 1;

            float3 d0;

            float3 s11 = make_float3(input[(isy1 * inputWidth + isx1) * inputChannels + 0] ,
                    input[(isy1 * inputWidth + isx1) * inputChannels + 1] ,
                    input[(isy1 * inputWidth + isx1) * inputChannels + 2]);

            float3 s12 = make_float3(input[(isy1 * inputWidth + isx2) * inputChannels + 0] ,
                    input[(isy1 * inputWidth + isx2) * inputChannels + 1] ,
                    input[(isy1 * inputWidth + isx2) * inputChannels + 2]);

            float3 s21 = make_float3(input[(isy2 * inputWidth + isx1) * inputChannels + 0] ,
                    input[(isy2 * inputWidth + isx1) * inputChannels + 1] ,
                    input[(isy2 * inputWidth + isx1) * inputChannels + 2]);

            float3 s22 = make_float3(input[(isy2 * inputWidth + isx2) * inputChannels + 0] ,
                    input[(isy2 * inputWidth + isx2) * inputChannels + 1] ,
                    input[(isy2 * inputWidth + isx2) * inputChannels + 2]);


            float h_rst00, h_rst01;
            // B
            if( dy > xmax - 1)
            {
                h_rst00 = s11.x;
                h_rst01 = s21.x;
            }
            else
            {
                h_rst00 = s11.x * cbufx.x + s12.x * cbufx.y;
                h_rst01 = s21.x * cbufx.x + s22.x * cbufx.y;
            }
            // d0.x = h_rst00 * (1 - fy) + h_rst01 * fy;
            d0.x = h_rst00 * cbufy.x + h_rst01 * cbufy.y;

            // G
            if( dy > xmax - 1)
            {
                h_rst00 = s11.y;
                h_rst01 = s21.y;
            }
            else
            {
                h_rst00 = s11.y * cbufx.x + s12.y * cbufx.y;
                h_rst01 = s21.y * cbufx.x + s22.y * cbufx.y;
            }
            // d0.y = h_rst00 * (1 - fy) + h_rst01 * fy;
            d0.y = h_rst00 * cbufy.x + h_rst01 * cbufy.y;

            // R
            if( dy > xmax - 1)
            {
                h_rst00 = s11.z;
                h_rst01 = s21.z;
            }
            else
            {
                h_rst00 = s11.z * cbufx.x + s12.z * cbufx.y;
                h_rst01 = s21.z * cbufx.x + s22.z * cbufx.y;
            }
            // d0.z = h_rst00 * (1 - fy) + h_rst01 * fy;
            d0.z = h_rst00 * cbufy.x + h_rst01 * cbufy.y;

            /****************************/
            if( dy > xmax - 1)
            {
                h_rst00 = s11.x;
                h_rst01 = s21.x;
            }
            else
            {
                h_rst00 = s11.x * cbufx.x + s12.x * cbufx.y;
                h_rst01 = s21.x * cbufx.x + s22.x * cbufx.y;
            }
            // d0.x = h_rst00 * (1 - fy) + h_rst01 * fy;
            d0.x = h_rst00 * cbufy.x + h_rst01 * cbufy.y;
            /****************************/

            //output[(dy*outputWidth + dx) * 3 + 0 ] = (d0.x - mean_value.x)*scale; // B
            //output[(dy*outputWidth + dx) * 3 + 1 ] = (d0.y - mean_value.y)*scale; // G
            //output[(dy*outputWidth + dx) * 3 + 2 ] = (d0.z - mean_value.z)*scale; // R
            //printf("%f %f %f\n", (d0.x - mean_value.x)*scale, (d0.y - mean_value.x)*scale, (d0.z - mean_value.x)*scale);

            //color_type 
            //1:RGB
            //else:BGR
            float3 d_color;
            if (color_type==1)
            {   
                d_color.x=d0.z;
                d_color.y=d0.y;
                d_color.z=d0.x;
            }
            else
            {
                d_color=d0;
            }

            output[0*outputWidth*outputHeight + dy*outputWidth + dx] = (d_color.x - mean_value.x)*scale.x; 
            output[1*outputWidth*outputHeight + dy*outputWidth + dx] = (d_color.y - mean_value.y)*scale.y; 
            output[2*outputWidth*outputHeight + dy*outputWidth + dx] = (d_color.z - mean_value.z)*scale.z; 
        }
    }
}

// cudaPreImageNetMean
cudaError_t cudaPreImageScaleMean(unsigned char* input, int inputHeight, int inputWidth, int inputChannels,
                                  float* output, int outputHeight, int outputWidth, float3 scale, const float3& mean_value,const int color_type)
{
    const int inputBytes = inputHeight * inputWidth * inputChannels * sizeof(unsigned char);

    unsigned char *d_input;

    CHECK(cudaMalloc<unsigned char>(&d_input, inputBytes));

    CHECK(cudaMemcpy(d_input, input, inputBytes, cudaMemcpyHostToDevice));
//    printf("*********");
//    std::cout<<("*********")<<std::endl;
//    for(int i=0; i< 100; i++)
//    {
//        std::cout<<int(input[i])<<" ";
//    }
//    std::cout<<int(input[19999])<<" ";
//    std::cout<<int(input[20001])<<" ";
//    std::cout<<std::endl;
    //Specify a reasonable block size
    const dim3 block(16, 16);

    //Calculate grid size to cover the whole image
    const dim3 grid((outputWidth + block.x - 1) / block.x, (outputHeight + block.y - 1) / block.y);

    //Launch the size conversion kernel
    gpuPreImageScaleMean<<<grid, block>>>(d_input, inputWidth, inputHeight, inputChannels,
                                        output, outputWidth, outputHeight, scale, mean_value,color_type);
    

    //CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(d_input));

    return CUDA(cudaGetLastError());
}

// cudaPreImageNetMean
cudaError_t cudaPreImageScaleMeanV2(unsigned char* input, int inputHeight, int inputWidth, int inputChannels,
                                  float* output, int outputHeight, int outputWidth, float3 scale, const float3& mean_value,const int color_type)
{
    //std::cout<<("*********")<<std::endl;
    //const int inputBytes = inputHeight * inputWidth * inputChannels * sizeof(unsigned char);
    //CHECK(cudaMalloc<unsigned char>(&d_input, inputBytes));
    //unsigned char *cpu_data;
    //cpu_data=(unsigned char *)malloc(inputBytes);
    //CHECK(cudaMemcpy(cpu_data, input, inputBytes, cudaMemcpyDeviceToHost));

    //for(int i=0; i< 100; i++)
    //{
    //    std::cout<<int(cpu_data[i])<<" ";
    //}
    //std::cout<<std::endl;
    //Specify a reasonable block size
    
    const dim3 block(16, 16);

    //Calculate grid size to cover the whole image
    const dim3 grid((outputWidth + block.x - 1) / block.x, (outputHeight + block.y - 1) / block.y);

    //Launch the size conversion kernel
    gpuPreImageScaleMean<<<grid, block>>>(input, inputWidth, inputHeight, inputChannels,
            output, outputWidth, outputHeight, scale, mean_value,color_type);

    //CHECK(cudaDeviceSynchronize());

    return CUDA(cudaGetLastError());
}


// gpuPreImageNetMean
__global__ void gpuPreImageMean(float* output, size_t width, size_t height, float scale, float3 mean_value)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = width * height;

    float3 px  = make_float3(output[y * width + x + 0], output[y * width + x + 1], output[y * width + x + 2]);

    float3 bgr = make_float3((px.x - mean_value.x)*scale, (px.y - mean_value.y)*scale, (px.z - mean_value.z)*scale);

    output[n * 0 + y * width + x] = bgr.x;
    output[n * 1 + y * width + x] = bgr.y;
    output[n * 2 + y * width + x] = bgr.z;
}

// cudaPreImageNetMean
cudaError_t cudaPreImageMean(float* output, size_t width, size_t height, float scale, float3& mean_value)
{
    if( !output )
        return cudaErrorInvalidDevicePointer;

    if( width == 0 || height == 0)
        return cudaErrorInvalidValue;

    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(width, blockDim.x), iDivUp(height, blockDim.y));
    gpuPreImageMean<<<gridDim, blockDim>>>(output, width, height, scale, mean_value);

    cudaDeviceSynchronize();

    return CUDA(cudaGetLastError());
}

// cudaCropImage
__global__ void corpKernel(const unsigned char* input, int inputWidth, int inputChannels, unsigned char* output, int x, int y, int w, int h)
{
    const int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy = blockIdx.y * blockDim.y + threadIdx.y;

    const int outputWidth = w;

    if (dx >= x && dy >= y && dx < (x+w) && dy < (y+h))
    {
	  if (inputChannels == 3)
	  {
	   output[(outputWidth*(dy-y) + (dx-x))*3 + 0] = input[(inputWidth*dy + dx)*3 + 0];
	   output[(outputWidth*(dy-y) + (dx-x))*3 + 1] = input[(inputWidth*dy + dx)*3 + 1];
	   output[(outputWidth*(dy-y) + (dx-x))*3 + 2] = input[(inputWidth*dy + dx)*3 + 2];
	  }
	  
	  if (inputChannels == 4)
	  {
	   output[(outputWidth*(dy-y) + (dx-x))*4 + 0] = input[(inputWidth*dy + dx)*4 + 0];
	   output[(outputWidth*(dy-y) + (dx-x))*4 + 1] = input[(inputWidth*dy + dx)*4 + 1];
	   output[(outputWidth*(dy-y) + (dx-x))*4 + 2] = input[(inputWidth*dy + dx)*4 + 2];
	   output[(outputWidth*(dy-y) + (dx-x))*4 + 3] = input[(inputWidth*dy + dx)*4 + 3];
	  }
    }
}

cudaError_t cudaCropImage(const unsigned char* input, int inputWidth, int inputHeight, int inputChannels,
        unsigned char* output, int x, int y, int w, int h)
{
    if( !output )
        return cudaErrorInvalidDevicePointer;

    assert(inputChannels == 3 || inputChannels == 4);  // BGR or BGRA


    assert(x >= 0 && y >= 0 && (x+w) <= inputWidth && (y+h) <= inputHeight);

    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(inputWidth, blockDim.x), iDivUp(inputHeight, blockDim.y));

    corpKernel<<<gridDim, blockDim>>>(input, inputWidth, inputChannels, output, x, y, w, h);

    //if(output == NULL){ 
    //    std::cout<<"OUTPUT is NULL"<<std::endl;
    //}else{
    //    std::cout<<"OUTPUT is NOTNULL"<<std::endl;
    //}

    //cudaDeviceSynchronize();

    return CUDA(cudaGetLastError());
}
#if 0
// cudaCropImage

__global__ void corpKernel(const unsigned char* input, int inputWidth, unsigned char* output, int x1, int y1, int x2, int y2)
{
    const int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy = blockIdx.y * blockDim.y + threadIdx.y;

    const int outputWidth = x2 - x1;

    if (dx >= x1 && dy >= y1 && dx <= x2 && dy <= y2)
    {
        output[outputWidth*(dy-y1) + (dx-x1) + 0] = input[inputWidth*dy + dx + 0];
        output[outputWidth*(dy-y1) + (dx-x1) + 1] = input[inputWidth*dy + dx + 1];
        output[outputWidth*(dy-y1) + (dx-x1) + 2] = input[inputWidth*dy + dx + 2];
    }
}

cudaError_t cudaCropImage(const unsigned char* input, int inputWidth, int inputHeight, int inputChannels,
        unsigned char* output, int x1, int y1, int x2, int y2)
{
    if( !output )
        return cudaErrorInvalidDevicePointer;

    assert(inputChannels == 3 || inputChannels == 4);  // BGR or BGRA

    assert( x2 > x1 && y2> y1);

    assert(x1>=0 && y1>=0 && x2<=inputWidth && y2<=inputHeight);

    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(inputWidth, blockDim.x), iDivUp(inputHeight, blockDim.y));

    corpKernel<<<gridDim, blockDim>>>(input, inputWidth, output, x1, y1, x2, y2);

    //cudaDeviceSynchronize();

    return CUDA(cudaGetLastError());
}
#endif
}
