#ifndef CUDA_H
#define CUDA_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "util.h"



__device__ float Relu(float);
void CudaProp();
struct Params{
    int n;//datasize
    float * dense_1_8;
    float * dense_8_4;
    flaot * dense_4_1;
};
#endif
