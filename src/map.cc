#include "map.h"
#include<locale>
using namespace std;
void ToBucket(KeysLogits *input, InitResultGpu *initResultGpu)
{
    int currentSize = 0;
    for (int i = 0; i < input->sizeKeys; i++)
    {
        int pos = int(round(*(input->logits + i)));
        if (*(initResultGpu->sortedList + pos) == 0)
        {
            *(initResultGpu->sortedList + pos) = *(input->keys + i);//*(input->logits+i);
        }
        else
        {
            if (currentSize == 0)
            {
                initResultGpu->waitedList = (double *)malloc(sizeof(double));
            }
            else
            {
                initResultGpu->waitedList = (double *)realloc(initResultGpu->waitedList, sizeof(double) * (currentSize + 1));
            }
            *(initResultGpu->waitedList + currentSize) = *(input->keys + i);
            currentSize++;
        }
    }
    initResultGpu->sizeWaited = currentSize;
}
void Merge(InitResultGpu *initResult, double *finalResult)
{
    cout<<endl<<"start merge"<<endl;
    int indexa = 0;
    int indexb = 0;
    int indexc = 0;
    while (initResult->sizeSorted > indexa && initResult->sizeWaited > indexb)
    {
        if (*(initResult->sortedList + indexa) != 0)
        {

            if (*(initResult->sortedList + indexa) <= *(initResult->waitedList + indexb))
            {
                *(finalResult + indexc++) = *(initResult->sortedList + indexa++);
            }
            else
            {
                *(finalResult + indexc++) = *(initResult->waitedList + indexb++);
            }
        }
        else
        {
            ++indexa;
        }
    }
    
    for (int i = indexa; i < initResult->sizeSorted; i++){
        if(*(initResult->sortedList+i)!=0){
            *(finalResult + indexc++) = *(initResult->sortedList + i);
        }
    }
    for (int i = indexb; i < initResult->sizeWaited; i++)
        *(finalResult + indexc++) = *(initResult->waitedList + i);
}
void Certify(double * array, int dataSize){
    cout<<"in certify"<<endl;
    double max_ = 0;
    int th = 0;
    for (int i = 0; i < dataSize; ++i){
        if(*(array+i) < max_){
 //           cout<<"disorder because"<<i<<"-th value:"<<*(array+i)<<"    <   "<<th<<"th:"<<"max_:"<<max_<<endl;
            return;
        }else{
        ++th;
            max_ = *(array + i);
        }
    }
    cout<<"sorted"<<endl;
}
