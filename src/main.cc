#include "cuda.h"
#include "quick.h"
#include "util.h"
#include "map.h"
#include<cmath>
using namespace std;

int main()
{
   int dataSize = 32;
   float *rawData = Initialize(dataSize, 1);
   KeysLogits *keysLogits = (KeysLogits *)malloc(sizeof(KeysLogits));
   keysLogits->sizeKeys = dataSize;
   keysLogits->sizeLogits = dataSize;
   keysLogits->keys = rawData;
   // keysLogits->logits=(float*)malloc(sizeof(float)*(keysLogits->sizeLogits));

   float max=model(keysLogits, dataSize, rawData);
   cout << "***********************logist***********************" << endl;
   for (int i = 0; i < dataSize; i++)
   {
      cout << *(keysLogits->logits + i) << "      ";
   }
   cout << endl
        << "max:" << max;


   InitResultGpu *initResultGpu=(InitResultGpu*)malloc(sizeof(initResultGpu));
   initResultGpu->sizeSorted=int(round(max))+1;
   cout << endl
        << "length:" <<initResultGpu->sizeSorted;
   initResultGpu->sizeWaited=0;
   initResultGpu->sortedList=(float*)malloc(sizeof(float)*initResultGpu->sizeSorted);
   for(int i=0;i<initResultGpu->sizeSorted;i++)
      *(initResultGpu->sortedList+i)=FLT_MAX;

   ToBucket(keysLogits,initResultGpu);
   cout<<endl<<"waited size:"<<initResultGpu->sizeWaited<<endl;

   float* sortResult=(float*)malloc(sizeof(float)*dataSize);
   Merge(initResultGpu,sortResult);
   // cout<<"************************final result"<<endl;
   // for(int i=0;i<dataSize;i++)
   // {
   //    cout<<*(sortResult+i)<<"  ";
   // } 
   
   free(initResultGpu->sortedList);
   free(initResultGpu);
   free(keysLogits->keys);
   free(keysLogits->logits);
   free(keysLogits);
   free(sortResult);
}
