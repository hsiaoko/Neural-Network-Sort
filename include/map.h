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
//初步排序
void ToBucket(Eigen::MatrixXd*,InitResut*);
//合并
void Merge(InitResut*,vector<double>*);

#endif
