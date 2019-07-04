#include"cuda.h"
#include"util.h"
#include<iomanip>
#include<iostream>
using namespace std;
#define max(a, b) (a > b ? a : b)
#define relu(a) (a > 0 ? a : 0)
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
__device__ int GetThreadX(){
    return blockIdx.x*blockDim.x+threadIdx.x;
}
__device__ int GetThreadY()
{
    return blockIdx.y*blockDim.y+threadIdx.y;
}

__global__ void Dense_2D_2D(double *AD, double * BD, double * bias, Dimension * dim, double * outputD){
    int tid_x = GetThreadX();//threadIdx.x + blockDim.x*blockIdx.x;
    int tid_y = GetThreadY();//blockIdx.y*blockDim.y+ threadIdx.y;
    while(tid_x < dim->d1){
        while (tid_y < dim->d3){
            double tmpValue = 0;
            for (int i = 0; i< dim->d2;++i){
                tmpValue += *(AD + tid_x *dim->d2 + i) * (*(BD + tid_y +i*dim->d3));
            }
            *(outputD + tid_x*dim->d3 + tid_y) = relu(tmpValue+*(bias+tid_y));
            tmpValue = 0;
            tid_y += blockDim.y * gridDim.y;
        }
        tid_y = blockIdx.y*blockDim.y+ threadIdx.y;
        tid_x += blockDim.x*gridDim.x;
    }
}
__global__ void MAX(double * arrayD, double *outputD, int threadPerBlock, int dataSize){
    __shared__ double cache[512];
    int cacheIndex = threadIdx.x;
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    double tmpMax = 0;
    while(tid < dataSize){
        tmpMax = *(arrayD+tid);//max(*(arrayD+tid), tmpMax);
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = tmpMax;
    
    __syncthreads();
    int i = blockDim.x/2;
    while(i !=0){
        if(cacheIndex < i){
            tmpMax = max( *(cache+cacheIndex),  *(cache+cacheIndex+i));
            cache[cacheIndex] = tmpMax;
        }
        __syncthreads();
            i/=2;
    }
    if(cacheIndex == 0){
        *(outputD+blockIdx.x) = tmpMax;
    }

}
double MAX_1D(double *inputD, int dataSize){
    int threadPerBlock = 512;
    int blocksPerGrid  = (dataSize+threadPerBlock)/threadPerBlock;
    cout<<"blockPerGrid:"<<blocksPerGrid<<endl;
    double *input;
    input = (double*)malloc(sizeof(double)*dataSize);

    cudaMemcpy(input, inputD, sizeof(double)*dataSize, cudaMemcpyDeviceToHost);

    double *maxOutput_d, *maxOutput;
    cudaMalloc((void**)&maxOutput_d,sizeof(double)*blocksPerGrid);
    maxOutput = (double*)malloc(sizeof(double)*blocksPerGrid);

    MAX<<<blocksPerGrid, threadPerBlock>>>(inputD, maxOutput_d, threadPerBlock, dataSize);

    cudaMemcpy(maxOutput, maxOutput_d, sizeof(double)*blocksPerGrid, cudaMemcpyDeviceToHost);
 //   cout<<"MAX:"<<*(maxOutput)<<endl;
    int maxValue = 0;
    for (int i = 0 ; i < blocksPerGrid; ++i){
        maxValue = max(*(maxOutput+i),maxValue);
    }
   // cout<<"MAX in MAX_1D:"<<maxValue<<endl;
    return maxValue;
}

double model(KeysLogits*keysLogits,int dataSize,double*rawData){
    cout<<setprecision(6);
    int paramsSize=8;

    Dimension *dim=(Dimension*)malloc(sizeof(Dimension));
    dim->d1=dataSize;
    dim->d2=1;
    dim->d3=paramsSize;
    cout<<"dataSize:"<<dim->d1<<endl;;                               

    double * weights_1_8, *weights_8_4, *weights_4_1, *bias_1_8, *bias_8_4, *bias_4_1;
    double*input=rawData;
    double*output=(double*)malloc(sizeof(double)*dataSize*paramsSize);
	weights_1_8 = (double*)malloc(sizeof(double)*1*8);
	weights_8_4 = (double*)malloc(sizeof(double)*8*4);
	weights_4_1 = (double*)malloc(sizeof(double)*4*1);
	bias_1_8 = (double*)malloc(sizeof(double)*8);
	bias_8_4 = (double*)malloc(sizeof(double)*4);
    bias_4_1 = (double*)malloc(sizeof(double)*1);


    cout<<setprecision(8); 
    initializeWeightsAndBias(weights_1_8, bias_1_8, weights_8_4, bias_8_4, weights_4_1, bias_4_1);
    double *inputD,*paramsD,*biasD,*outputD;
    Dimension*dimD;
    cudaMalloc((void **)&inputD, sizeof(double) * dataSize);
    cudaMalloc((void **)&paramsD, sizeof(double) * paramsSize);
    cudaMalloc((void **)&biasD, sizeof(double) * paramsSize);
    cudaMalloc((void **)&outputD, sizeof(double) * dataSize*paramsSize);
    cudaMalloc((void**)&dimD,sizeof(Dimension));

    cudaMemcpy( inputD, input, sizeof(double) * dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy( paramsD, weights_1_8, sizeof(double) * paramsSize, cudaMemcpyHostToDevice);
    cudaMemcpy( biasD, bias_1_8, sizeof(double) * paramsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dimD,dim,sizeof(Dimension),cudaMemcpyHostToDevice);
    dim3 grid(16384,8);
    dim3 block(1024,1);

    steady_clock::time_point Start = steady_clock::now();
    Dense_2D_2D<<<grid,block>>>(inputD,paramsD,biasD,dimD,outputD);
    
    Dimension *dim2=(Dimension*)malloc(sizeof(Dimension));
    dim2->d1=dataSize;
    dim2->d2=8;
    dim2->d3=4;

    double*params2D;
    double*bias2D;
    Dimension *dimD2;
    cudaMalloc((void **)&dimD2, sizeof(Dimension));
    cudaMalloc((void **)&params2D, sizeof(double) * dim2->d2*dim2->d3);
    cudaMalloc((void **)&bias2D, sizeof(double) * dim2->d3);
    cudaMemcpy(dimD2,dim2,sizeof(Dimension),cudaMemcpyHostToDevice);
    cudaMemcpy(params2D,weights_8_4,sizeof(double)*dim2->d2*dim2->d3,cudaMemcpyHostToDevice);
    cudaMemcpy(bias2D,bias_8_4,sizeof(double)*dim2->d3,cudaMemcpyHostToDevice);
    double*output2=(double*)malloc(sizeof(double)*dim2->d1*dim2->d3);
    double*output2D;
    cudaMalloc((void **)&output2D, sizeof(double) * dim2->d1*dim2->d3);

    Dense_2D_2D<<<grid,block >>>(outputD,params2D,bias2D,dimD2,output2D);
    dim->d1=dataSize;
    dim->d2=4;
    dim->d3=1;
    Dimension *dimD3;
    double *final=(double*)malloc(sizeof(double)*dim->d1*dim->d3);

    double *finalD;
    double*params3D;
    double*bias3D;
    cudaMalloc((void **)&finalD, sizeof(double) * dim->d1*dim->d3);
    cudaMalloc((void **)&params3D, sizeof(double) * dim->d2);
    cudaMalloc((void **)&bias3D, sizeof(double) * dim->d3);
    cudaMalloc((void **)&dimD3, sizeof(Dimension));

    cudaMemcpy( params3D, weights_4_1, sizeof(double) * dim->d2, cudaMemcpyHostToDevice);
    cudaMemcpy(dimD3,dim,sizeof(Dimension),cudaMemcpyHostToDevice);
    cudaMemcpy(bias3D,bias_4_1,sizeof(double)*dim->d3,cudaMemcpyHostToDevice);

    Dense_2D_2D<<<grid,block>>>(output2D,params3D,bias3D,dimD3,finalD);
    cudaMemcpy(final,finalD,sizeof(double)*dim->d1*dim->d3,cudaMemcpyDeviceToHost);
    steady_clock::time_point nonMax = steady_clock::now();
    double max=MAX_1D(finalD, dim->d1);
    
    steady_clock::time_point end = steady_clock::now();
    duration<double, std::milli> *timePredicte = new duration<double, std::milli>(end -Start);
    duration<double, std::milli> *timeNonMax = new duration<double, std::milli>(nonMax -Start);
    cout <<endl<< "consumming of predict:" << timePredicte->count() << " ms" << endl;
    cout <<endl<< "consumming of nonMax:" << timeNonMax->count() << " ms" << endl;

    keysLogits->logits=final;
    // free(input);
    free(output);
    free(dim);
    free(dim2);
    free(output2);
    // free(final);
    cudaFree(inputD);
    cudaFree(paramsD);
    cudaFree(biasD);
    cudaFree(outputD);
    cudaFree(dimD);
    cudaFree(dimD2);
    cudaFree(params2D);
    cudaFree(output2D);
    cudaFree(finalD);


    return max;
}


