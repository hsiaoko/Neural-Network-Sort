#include "cuda.h"
#include "quick.h"
#include "util.h"
#include "map.h"
#include <cmath>
using namespace std;

int main()
{
   int dataSize = 31457280;
   double *rawData = Initialize(dataSize, 4);

    steady_clock::time_point time1 = steady_clock::now();
    KeysLogits *keysLogits = (KeysLogits *)malloc(sizeof(KeysLogits));
    keysLogits->sizeKeys = dataSize;
    keysLogits->sizeLogits = dataSize;
    keysLogits->keys = rawData;

    double max = model(keysLogits, dataSize, rawData);


    InitResultGpu *initResultGpu = (InitResultGpu *)malloc(sizeof(initResultGpu));
    initResultGpu->sizeSorted = int(round(max)) + 1;
    initResultGpu->sizeWaited = 0;
    initResultGpu->sortedList = (double *)malloc(sizeof(double) * initResultGpu->sizeSorted);
    memset(initResultGpu->sortedList, 0, sizeof(double)*initResultGpu->sizeSorted);

    steady_clock::time_point time2 = steady_clock::now();

    ToBucket(keysLogits, initResultGpu);

    double max_ = 0;
    steady_clock::time_point time3 = steady_clock::now();
    
    QuickSort(initResultGpu->waitedList, 0, initResultGpu->sizeWaited-1);
    steady_clock::time_point time4 = steady_clock::now();
    
    
    double *sortResult = (double *)malloc(sizeof(double) * dataSize);

    Merge(initResultGpu, sortResult);
    steady_clock::time_point time5 = steady_clock::now();

    duration<double, std::milli> *timeModel = new duration<double, std::milli>(time2 - time1);
    duration<double, std::milli> *timeTobucket = new duration<double, std::milli>(time3 - time2);
    duration<double, std::milli> *timeQuickSort = new duration<double, std::milli>(time4 - time3);
    duration<double, std::milli> *timeMerge = new duration<double, std::milli>(time5 - time4);
    duration<double, std::milli> *timeNnsSort = new duration<double, std::milli>(time5 - time1);
   
    cout << "length of big array:" << initResultGpu->sizeSorted;
    cout << " waited size:" << initResultGpu->sizeWaited << endl;
    cout << "consumming of to model:" << timeModel->count() << " ms" << endl;
    cout << "consumming of bucket:" << timeTobucket->count() << " ms" << endl;
    cout << "consumming of quick:" << timeQuickSort->count() << " ms" << endl;
    cout << "consumming of merge:" << timeMerge->count() << " ms" << endl;
    cout << "consumming of nns:" << timeNnsSort->count() << " ms" << endl;

    Certify(sortResult, keysLogits->sizeKeys);
    free(initResultGpu->sortedList);
    free(initResultGpu);
    free(keysLogits->keys);
    free(keysLogits->logits);
    free(keysLogits);
    free(sortResult);
}
