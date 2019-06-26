#include "map.h"


void ToBucket(KeysLogits*input,InitResultGpu *initResultGpu)
{
    int currentSize=0;
    for(int i=0;i<input->sizeKeys;i++)
    {
        int pos = int(round(*(input->logits+i)));
        if (*(initResultGpu->sortedList+pos) == FLT_MAX)
        {
            *(initResultGpu->sortedList+pos) = *(input->keys+i);
        }
        else
        {
            if(currentSize==0)
            {
                initResultGpu->waitedList=(float*)malloc(sizeof(float));
            }
            else
            {
                initResultGpu->waitedList=(float*)realloc(initResultGpu->waitedList,sizeof(float));
            }
            *(initResultGpu->waitedList+currentSize)=*(input->keys+i);
            currentSize++;
        }
    }
    initResultGpu->sizeWaited=currentSize;
}

void Merge(InitResut *initResult, vector<double> *finalResult)
{
    int indexa = 0;
    int indexb = 0;
    while (initResult->sortedList->size() > indexa && initResult->waitedList->size() > indexb)
    {
        if ((*initResult->sortedList)[indexa]!=FLT_MAX)
        {
            if ((*initResult->sortedList)[indexa] <= (*initResult->waitedList)[indexb])
            {
                finalResult->push_back((*initResult->sortedList)[indexa]);
                indexa += 1;
            }
            else
            {
                finalResult->push_back((*initResult->waitedList)[indexb]);
                indexb += 1;
            }
        }
        else
        {
            ++indexa;
        }
        
    }
    finalResult->insert(finalResult->end(), initResult->sortedList->begin() + indexa, initResult->sortedList->end());
    finalResult->insert(finalResult->end(), initResult->waitedList->begin() + indexb, initResult->waitedList->end());
    cout << "final:" << finalResult->size() << endl;
}
void Merge(InitResultGpu * initResult, float * finalResult){

    int indexa = 0;
    int indexb = 0;
    int indexc = 0;
    while(initResult->sizeSorted > indexa && initResult->sizeWaited > indexb){
        if (*(initResult->sortedList+indexa)!=FLT_MAX){
            
            if(*(initResult->sortedList+indexa)<= *(initResult->waitedList+indexb)){
                *(finalResult+indexc++) = *(initResult->sortedList+indexa++);
            }else{
                *(finalResult+indexc++) = *(initResult->waitedList+indexb++);
            }
        }else{
            ++indexa;
        }
        cout<<"indexa:"<<indexa<<" indexb:"<<indexb<<" indexc:"<<indexc<<endl;
    }
    for(int i=indexa;i<initResult->sizeSorted;i++)
         *(finalResult+indexc++) = *(initResult->sortedList+i);

    for(int i=indexb;i<initResult->sizeWaited;i++)
         *(finalResult+indexc++) = *(initResult->waitedList+i);
}
