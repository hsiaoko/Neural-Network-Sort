#ifndef CUDA_H
#define CUDA_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "util.h"
#define NUMSPERBLOCk 1024;
#define THREADLIMIT 67107840;
#define DOWNLIMIT 1;
#define UPLIMIT 128;


__device__ float Relu(float);
void CudaProp();
struct Params{
    int n;//datasize
    float * dense_1_8;
    float * dense_8_4;
    float * dense_4_1;
};
struct Dimension{
    int d1;
    int d2;
    int d3;
};
__device__ int GetThreadX();
__device__ int GetThreadY();
__global__ void Dense2D2D( float * , float *,Dimension*,float* );
__global__ void Dense_1_8(float*,float*,float*,Dimension*,float*);
__global__ void Dense_4_1(float*,float*,float*,Dimension*,float*);
__global__ void max_2D_1D (int , float * , float * );
int max_1D(float * , int );
#endif
