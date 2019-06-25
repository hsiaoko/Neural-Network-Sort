#include "map.h"

void ToBucket(Eigen::MatrixXd *input, InitResut *initResult)
{
    for (int i = 0; i < input->rows(); i++)
    {
        int pos = int(round((*input)(i, 1))*2);
        if ((*initResult->sortedList)[pos] == FLT_MAX)
        {
            (*initResult->sortedList)[pos] = (*input)(i, 0);
        }
        else
        {
            initResult->waitedList->push_back((*input)(i, 0));
        }
    }
    // cout<<"-------------------------sorted--------------------\n";
    // for (int i = 0; i < initResult->sortedList->size(); i++)
    // {
    //     cout << (*initResult->sortedList)[i] << endl;
    // }
    // cout<<"-------------------------waited--------------------\n";
    // for (int j = 0; j < initResult->waitedList->size(); j++)
    // {
    //     cout << (*initResult->waitedList)[j] << endl;
    // }
    cout << "size of waited:" << initResult->waitedList->size() << endl;
    cout << "size of order elements:" << input->rows() - initResult->waitedList->size() << endl;
    cout << "size of array:" << initResult->sortedList->size() << endl;
}

void ToBucket(keys_logits*input,InitResultGpu *initResultGpu)
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
    int indexa = 0;
    int indexc = 0
    while(initResult->sizeSorted > indexa && initResult->waitedList->sizeWaited > indexb){
        if (initResult->sortedList[indexa]!=FLT_MAX){
            if(initResult->sortedList[indexa]<= initResult->waitedList[indexb]){
                finalResult[indexc++] = initResult->sortedList[indexa];
                ++indexa;
            }else{
                finalResult[indexc++] = initResult->sortedList[indexa];
                ++indexb;

            }
        }else{
            ++indexa;
        }
    }
}
