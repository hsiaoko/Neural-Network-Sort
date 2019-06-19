#ifndef GRAPH_H
#define GRAPH_H
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include"util.h"
#include "map.h"
using namespace std;
using namespace Eigen;

MatrixXf relu_(MatrixXf );
MatrixXf max_out_(MatrixXf , int );
MatrixXf *graph_(MatrixXf *, int , MatrixXf *);

#endif