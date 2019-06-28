#include"cuda.cuh"
#include "quick.h"
#include"util.h"
#include"map.h"
using namespace std;

void model(){
    int dataSize=32;
    int paramsSize=8;
    
    float*input=Initialize(dataSize,1);
    float*params=(float*)malloc(sizeof(float)*paramsSize);
    float*bias=(float*)malloc(sizeof(float)*paramsSize);
    float*output=(float*)malloc(sizeof(float)*dataSize*paramsSize);
    Dimension *dim=(Dimension*)malloc(sizeof(Dimension));
    dim->d1=dataSize;
    dim->d2=1;
    dim->d3=paramsSize;                                                                                                                                                                                                         ;
    for(int i=0;i<paramsSize;i++)
    {
        *(params+i)=2;
        *(bias+i)=0.1;
    }
    cout<<"***********************input***********************"<<endl;
    for(int i=0;i<dataSize;i++)
    {
        cout<<*(input+i)<<' ';
        
    }
    cout<<endl;
    cout<<"***********************params***********************"<<endl;
    for(int i=0;i<paramsSize;i++)
    {
        cout<<*(params+i)<<' ';
    }
    cout<<endl;
    cout<<"***********************bias***********************"<<endl;
    for(int i=0;i<paramsSize;i++)
    {
        cout<<*(bias+i)<<' ';
    }
    cout<<endl;
    float *inputD,*paramsD,*biasD,*outputD;
    Dimension*dimD;
    cudaMalloc((void **)&inputD, sizeof(float) * dataSize);
    cudaMalloc((void **)&paramsD, sizeof(float) * paramsSize);
    cudaMalloc((void **)&biasD, sizeof(float) * paramsSize);
    cudaMalloc((void **)&outputD, sizeof(float) * dataSize*paramsSize);
    cudaMalloc((void**)&dimD,sizeof(Dimension));

    cudaMemcpy( inputD, input, sizeof(float) * dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy( paramsD, params, sizeof(float) * paramsSize, cudaMemcpyHostToDevice);
    cudaMemcpy( biasD, bias, sizeof(float) * paramsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dimD,dim,sizeof(Dimension),cudaMemcpyHostToDevice);
    dim3 grid(4,2);
    dim3 block(2,2);
    Dense_1_8<<<1,block>>>(inputD,paramsD,biasD,dimD,outputD);

    cudaMemcpy( output, outputD, sizeof(float) * dataSize*paramsSize, cudaMemcpyDeviceToHost);

    cout<<"***********************result***********************"<<endl;
    for(int i=0;i<dataSize*paramsSize;i++)
    {
        cout<<*(output+i)<<"      ";
        if((i+1)%paramsSize==0)
            cout<<endl;
    }



    Dimension *dim2=(Dimension*)malloc(sizeof(Dimension));
    dim2->d1=dataSize;
    dim2->d2=8;
    dim2->d3=4;

    float*params2=(float*)malloc(sizeof(float)*dim2->d2*dim2->d3);
    initializeMatrix(params2);
    float*params2D;
    Dimension *dimD2;
    cudaMalloc((void **)&dimD2, sizeof(Dimension));
    cudaMalloc((void **)&params2D, sizeof(float) * dim2->d2*dim2->d3);
    cudaMemcpy(dimD2,dim2,sizeof(Dimension),cudaMemcpyHostToDevice);
    cudaMemcpy(params2D,params2,sizeof(float)*dim2->d2*dim2->d3,cudaMemcpyHostToDevice);
    float*output2=(float*)malloc(sizeof(float)*dim2->d1*dim2->d3);
    float*output2D;
    cudaMalloc((void **)&output2D, sizeof(float) * dim2->d1*dim2->d3);
    cout<<"***********************param2***********************"<<endl;
    for(int i=0;i<dim2->d2*dim2->d3;i++)
    {
        cout<<*(params2+i)<<"      ";
        if((i+1)%dim2->d3==0)
            cout<<endl;
    }
    dim3 grid2(2,2);
    dim3 block2(2);
    Dense2D2D<<<grid2,block2 >>>(outputD,params2D,dimD2,output2D);
    cudaMemcpy( output2, output2D, sizeof(float) * dim2->d1*dim2->d3, cudaMemcpyDeviceToHost);
    cout<<"***********************result2***********************"<<endl;
    for(int i=0;i<dim2->d1*dim2->d3;i++)
    {
        cout<<*(output2+i)<<"      ";
        if((i+1)%dim2->d3==0)
            cout<<endl;
    }
    










    // Dimension *dim33=(Dimension*)malloc(sizeof(Dimension));
    // Dimension *dim33D;
    dim->d1=dataSize;
    dim->d2=4;
    dim->d3=1;
    // Dimension *dimD2;
    float *final=(float*)malloc(sizeof(float)*dim->d1*dim->d3);
    float *finalD;
    cudaMalloc((void **)&finalD, sizeof(float) * dim->d1*dim->d3);
    cudaMalloc((void **)&dimD2, sizeof(Dimension));
    cudaMemcpy(dimD2,dim,sizeof(Dimension),cudaMemcpyHostToDevice);
    
    Dense_4_1<<<1,block>>>(output2D,paramsD,biasD,dimD2,finalD);
    cudaMemcpy( final, finalD, sizeof(float) * dim->d1*dim->d3, cudaMemcpyDeviceToHost);
    cout<<endl;
    cout<<"***********************result3***********************"<<endl;
    for(int i=0;i<dim->d1*dim->d3;i++)
    {
        cout<<*(final+i)<<"      ";
    }
    int max=max_1D(finalD, dim->d1);

    cout<<endl<<"max:"<<max<<endl;




    cout<<endl;
    free(input);
    free(params);
    free(bias);
    free(output);
    free(dim);
    cudaFree(inputD);
    cudaFree(paramsD);
    cudaFree(biasD);
    cudaFree(outputD);
    cudaFree(dimD);


    cout<<"hello world\n";
}

int main(){
   CudaProp(); 
    


}

