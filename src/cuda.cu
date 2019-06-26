#include"cuda.cuh"
#include"util.h"
void model(float *input,KeysLogits*keylogists)
{
    float *B = (float *)malloc(sizeof(float) * dense1);
    float *W2=(float *)malloc(sizeof(float) * dense2);
    float *C = (float *)malloc(sizeof(float) * data_size * 4);
    float *bias1=(float*)malloc(sizeof(float)*dense1);
    float *bias2=(float*)malloc(sizeof(float)*1);
    *(B) = 1;
    *(B+1) = 0;
    *(B+2) = 0;
    *(B+3) = 1;

    *(W2) = 1;
    *(W2+1) = 0;
    *(W2+2) = 0;
    *(W2+3) = 1;

    *(bias1) = 1;
    *(bias1+1) = 1;
    *(bias1+2) = 1;
    *(bias1+3) = 1;

    *bias2=1;
    steady_clock::time_point t1=steady_clock::now();


    float *A_d, *B_d,*W2_d,*C_d,*bias1_d,*bias2_d;

    cudaMalloc((void **)&A_d, sizeof(float) * data_size);
    cudaMalloc((void **)&B_d, sizeof(float) * dense1);
    cudaMalloc((void **)&C_d, sizeof(float) * data_size * 4);
    cudaMalloc((void **)&W2_d, sizeof(float) * dense2);
    cudaMalloc((void **)&bias1_d, sizeof(float) * dense1);
    cudaMalloc((void **)&bias2_d, sizeof(float) * 1);

    cudaMemcpy( A_d, input, sizeof(float) * data_size, cudaMemcpyHostToDevice);
    cudaMemcpy( B_d, B, sizeof(float) * dense1, cudaMemcpyHostToDevice);
    cudaMemcpy( W2_d, W2, sizeof(float) * dense2, cudaMemcpyHostToDevice);
    cudaMemcpy( bias1_d, bias1, sizeof(float) * dense1, cudaMemcpyHostToDevice);
    cudaMemcpy( bias2_d, bias2, sizeof(float) * 1, cudaMemcpyHostToDevice);


    dim3 grindSize(block_x,block_y);
    dim3 blockSize(threads_number_perblock);

    int total_threads = grindSize.x*grindSize.y*blockSize.x*blockSize.y;
    cout<<"total_threads:"<<total_threads<<endl;
    //int sectionSize = data_size/total_threads;
    

   


    dense_1_4 <<<grindSize, blockSize>>> (A_d, B_d,bias1_d, C_d);
     float *temp = (float *)malloc(sizeof(float) * data_size * 4);
     cudaMemcpy( temp, C_d, sizeof(float) * data_size * 4, cudaMemcpyDeviceToHost);
     cout<<"********************vector A********************";
     cout<<endl;
     for(int i=0;i<data_size;i++)
     {
         cout<<input[i]<<"    ";
     }
     cout<<endl;
     cout<<"********************matrixc C********************";
     cout<<endl;
     for(int i=0;i<data_size*4;i++)
     {
      cout<<*(temp+i)<<"    ";
      if((i+1)%4==0)
         cout<<endl;
     }
     cout<<endl;
    dense_4_1<<<grindSize, blockSize>>>(C_d,W2_d,bias2_d,A_d);
    steady_clock::time_point t2=steady_clock::now();
    cudaMemcpy( keylogists->logits, A_d, sizeof(float) * data_size, cudaMemcpyDeviceToHost);



    cout<<"********************vector B********************";
    cout<<endl;
    for(int i=0;i<dense1;i++)
    {
     cout<<*(B+i)<<' ';
    }
    cout<<endl;
    cout<<"********************vector logist********************";
    cout<<endl;
    for(int i=0;i<data_size;i++)
    {
     cout<<*(keylogists->logits+i)<<"    ";
    }
    cout<<endl;

    free(B);
    free(C);
    free(W2);
    free(bias1);
    free(bias2);
    free(temp);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaFree(bias1_d);
    cudaFree(W2_d);
    cudaFree(bias2_d);
    cout << "block_cover:" << block_cover<<endl;
    cout <<"data_size:"<<data_size<<endl;
    cout <<"total_threads"<<total_threads<<endl;
    duration<double,std::milli> *timeSpan = new duration<double,std::milli>(t2-t1);
    cout<<"consumming of alg:"<<timeSpan->count()<<" ms"<<endl;

}

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
__device__ float relu_(float);
struct Output{
    int thread_id;
    int block_id_x;
    int block_id_y;
    float A;
    float B;
    float result;
};
__device__ int getGlobalIdx_2D_1D(){
    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    int thread_id = block_id * (threads_number_perblock) + threadIdx.y * blockDim.x + threadIdx.x;
    return thread_id;
}
__global__ void dense_1_4(float *M, float *N,float*bias, float *P){
    int thread_id = getGlobalIdx_2D_1D();
    int start_point_thread = thread_id*thread_cover;
    int end_point_thread = (thread_id+1)*thread_cover;
    for (int i = start_point_thread; i< end_point_thread; ++i){
         for(int j=0;j<4;j++)
         {
            *(P + i*4+j)= relu_(M[i] * N[j]+bias[j]);
         }
    }

}
__global__ void dense_4_1(float *M,float*N,float*bias,float*P)
{
    int thread_id=getGlobalIdx_2D_1D();  
    int start_point_thread = thread_id*thread_cover;
    int end_point_thread = (thread_id+1)*thread_cover;
    for(int i=start_point_thread;i<end_point_thread;++i)
    {
        float temp=0;
        for(int j=0;j<4;++j)
        {
            temp+=M[i*4+j]*N[j];

        }
        *(P+i)=temp+(*bias);
    }
}

__device__ float relu_(float p){
    if (p > 0){
        return p;
    }else{
        return 0;
    }
}

