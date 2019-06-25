#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <sys/time.h> 
#include <stdio.h>
#include "util.h"
using std::cout;
using std::endl;

#define dense1 4
#define dense2 4
#define block_x 2
#define block_y 2
#define threads_number_perblock 1
#define block_cover data_size/(block_x *block_y)
#define thread_cover 2

#define data_size (block_x*block_y)*(threads_number_perblock)*thread_cover // (4*4) * (1024) * (64)  每个grind 有16个block ,每个block 有1024个线层，　每个线程处理1024个浮点数.



void CudaProp();
__device__ int getGlobalIdx_2D_1D();
__global__ void dense_1_4(float *, float *,float*, float *);
__global__ void dense_4_1(float *,float*,float*,float*);
__device__ float relu_(float );
void model();
