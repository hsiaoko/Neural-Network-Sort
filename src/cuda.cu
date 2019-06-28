#include"cuda.cuh"
#include"util.h"
void CudaProp()
{
    int device_count;
    cudaGetDeviceCount(&device_count);
    for (int i=0; i<device_count; ++i ){
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp,i);
        cout<<"使用GPU"<<i<<":"<<devProp.name<<endl;
        cout<<"SM的数量"<<devProp.multiProcessorCount<<endl;

        cout<< "每个线程块的共享内存大小" << devProp.sharedMemPerBlock / 1024.0  <<"KB" <<endl;
        cout<< "每个线程块的最大线程数"<< devProp.maxThreadsPerBlock<<endl;
        cout<<"设备上一个线程块可用的32位寄存器的数量"<< devProp.regsPerBlock <<endl;
        cout<<"每个EM的最大线程数"<< devProp.maxThreadsPerMultiProcessor<<endl;


        cout<< "设备上多处理器的数量" << devProp.multiProcessorCount<<endl;
        cout<< "设备上EM的最大线程束数" << endl;

        cout<<"=================================================================="<<endl;
    }
}

__device__ float Relu(float p){
    if (p > 0){
        return p;
    }else{
        return 0;
    }
}
__device__ int GetThreadX(){
    return blockIdx.x*blockDim.x+threadIdx.x;
}
__device__ int GetThreadY()
{
    return blockIdx.y*blockDim.y+threadIdx.y;
}
__global__ void Dense2D2D( float * input_d, float *matrix_d,Dimension*dim,float* output_d){
    //[m*n][m*k]
    int thread_cover=dim->d1/(gridDim.x*gridDim.y*blockDim.x);

    int blockId = gridDim.x * blockIdx.y+ blockIdx.x;
    int threadId = blockId * blockDim.x+ threadIdx.x;
    int start_point = threadId*thread_cover;
    int end_point = (threadId+1)*thread_cover;
    // (debugOutput+threadId)->start_point = start_point;
    // (debugOutput+threadId)->end_point = end_point;
    // (debugOutput+threadId)->threadId_x = threadId;


    // *(output_d)=*(matrix_d+10);
    for (int i = start_point; i< end_point; ++i){
        for (int j = 0; j < dim->d3; ++j){
            for (int k = 0; k < dim->d2; ++k){
                *(output_d+i*dim->d3+j) += *(input_d+i*dim->d2+k) * (*(matrix_d+j*dim->d2+k));
                // *(output_d+i*dim->d3+j)=66;

            }
        }
    }

}
__global__ void Dense_1_8(float*input,float*params,float*bias,Dimension*dim,float*output)
{

    int x=GetThreadX();
    int y=GetThreadY();
    while(x<dim->d1)
    {
        while(y<dim->d3)
        {
            *(output+x*(dim->d3)+y)=*(input+x)*(*(params+y))+(*(bias+y));
            y+=blockDim.y*gridDim.y;
        }
        y=GetThreadY();
        x+=blockDim.x*gridDim.x;
    }
}
__global__ void Dense_4_1(float*input,float*params,float*bias,Dimension*dim,float*output)
{
    int x=GetThreadX();
    int y=GetThreadY();
    while(x<dim->d1)
    {
        while(y<dim->d2)
        {
            *(input+x*(dim->d2)+y)*=(*(params+y));
            y+=blockDim.y*gridDim.y;
        }
        y=GetThreadY();
        x+=blockDim.x*gridDim.x;
    }
    x=GetThreadX();
    y=GetThreadY();
    int i=(blockIdx.x+blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y)+threadIdx.y*blockDim.x+threadIdx.x;
    while(i<dim->d1)
    {
        for(int j=0;j<dim->d2;j++)
            *(output+i)+=*(input+i*dim->d2+j);
        i+=gridDim.x*gridDim.y*blockDim.x*blockDim.y;
    }
    // int i=dim->d2/2;
    // while(x<dim->d1)
    // {
    //     while(i!=0)
    //     {
    //         while(y<i)
    //         {

    //             *(input+x*(dim->d2)+y)+=*(input+x*(dim->d2)+y+i);
    //             y+=blockDim.y*gridDim.y;
    //             __syncthreads();
    //         }
    //         i/=2;
    //     }
    //     y=GetThreadY();
    //     *(output+x)=*(input+x*(dim->d2))+(*(bias+y));
    //     x+=blockDim.x*gridDim.x;
    // }
}
__global__ void max_2D_1D (int dataSize, float * inputD, float * outputD){
    int blockId = gridDim.x * blockIdx.y+ blockIdx.x;
    int totalThreads = gridDim.x*gridDim.y * blockDim.x;
    int threadId = blockId * blockDim.x+ threadIdx.x;
    int threadCover = dataSize/(totalThreads);
    
    int startPoint = threadId*threadCover;
    int endPoint = (threadId+1)*threadCover;
    

    float tmpMax = 0;
    for (int i = startPoint; i < endPoint; ++i ){
        if (inputD[i] < tmpMax){
            continue;
        }else{
            tmpMax = inputD[i];
        }
    }
    *(outputD+threadId) = tmpMax;
}
int max_1D(float * input, int dataSize){
    // int max = 0;
    

    int gridDim_x = 2;
    int gridDim_y = 2;
    int blockDim_x = 2;
    int totalThreads = gridDim_x*gridDim_y*blockDim_x;


    dim3 gridSizeTmp(gridDim_x,gridDim_y);
    dim3 blockSizeTmp(blockDim_x);


    float *maxOutput_d, *maxOutput;
    cudaMalloc((void**)&maxOutput_d,sizeof(float)*totalThreads);
    maxOutput = (float*)malloc(sizeof(float)*totalThreads);

    // DebugOutput * debugOutput, *debugOutput_d;
    // debugOutput = (DebugOutput *)malloc(sizeof(DebugOutput)*totalThreads);
    // cudaMalloc((void**)&debugOutput_d,sizeof(DebugOutput)*totalThreads);
    
    max_2D_1D<<<gridSizeTmp, blockSizeTmp>>>(dataSize, input, maxOutput_d);
    cudaMemcpy(maxOutput, maxOutput_d,sizeof(float)*totalThreads,cudaMemcpyDeviceToHost);
    float tmpMax = 0;
    for (int i = 0; i< totalThreads; ++i){
        // cout<<"thread_i_max:"<<*(maxOutput+i)<<endl;
        if (*(maxOutput+i) <= tmpMax){
            continue;
        }else{

            tmpMax = *(maxOutput+i);
        }
    }


    cout<<"max:"<<tmpMax<<endl;
    return tmpMax;

}

