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

