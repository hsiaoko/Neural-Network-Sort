#ifndef MAP_H
#define MAP_H
#include<iostream>
#include<vector>
#include<Eigen/Dense>
#include<math.h>
#include<float.h>
using std::vector;
using std::cout;
using std::endl;
using namespace Eigen;
struct InitResut
{
    /* data */
    vector<double> *sortedList;
    vector<double> *waitedList;
};
struct InitResultGpu
{
    float*sortedList;
    float*waitedList;
}
//初步排序
void ToBucket(Eigen::MatrixXd*,InitResut*);
void ToBucket(float**,int,InitResultGpu*);
//合并
void Merge(InitResut*,vector<double>*);

#endif
