#ifndef MAP_H
#define MAP_H
#include<vector>
#include<cmath>
#include<locale>
#include"util.h"
#include<float.h>
#include<iostream>
#include<string.h>
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
    double*sortedList;
    double*waitedList;
};

//初步排序
void ToBucket(KeysLogits*,InitResultGpu*);
void Certify(double *, int);
//合并
//void Merge(InitResut*,vector<double>*);
void Merge(InitResultGpu*, double *);

#endif
