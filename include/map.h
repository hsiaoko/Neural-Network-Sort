#ifndef MAP_H
#define MAP_H
#include<vector>
#include<math.h>
#include<float.h>
#include"util.h"
struct InitResut
{
    /* data */
    vector<double> *sortedList;
    vector<double> *waitedList;
};
struct InitResultGpu
{
    int sizeSorted;
    int sizeWaited;
    float*sortedList;
    float*waitedList;
};

//初步排序
void ToBucket(KeysLogits*,InitResultGpu*);
//合并
void Merge(InitResut*,vector<double>*);
void Merge(InitResultGpu*, float *);

#endif
