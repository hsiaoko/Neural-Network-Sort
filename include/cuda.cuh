#ifndef CUDA_H
#define CUDA_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "util.h"



__device__ float Relu(float);
void CudaProp();
#endif
