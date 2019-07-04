#ifndef CUDA_H
#define CUDA_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "util.h"
#define NUMSPERBLOCk 1024;
#define THREADLIMIT 67107840;
#define DOWNLIMIT 1;
#define UPLIMIT 128;


__device__ double Relu(double);
void CudaProp();
struct Params{
    int n;//datasize
    double * dense_1_8;
    double * dense_8_4;
    double * dense_4_1;
};
struct Dimension{
    int d1;
    int d2;
    int d3;
};
__device__ int GetThreadX();
__device__ int GetThreadY();
__global__ void Dense_2D_2D(double *, double *, double *, Dimension *, double *);
__global__ void MAX(double * , double *, int, int );
double MAX_1D(double *inputD, int dataSize);
double model(KeysLogits*,int,double*);
#endif
