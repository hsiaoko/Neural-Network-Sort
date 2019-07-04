#ifndef UTIL_H
#define UTIL_H
#include <iostream>
#include <vector>
#include <random>
#include<locale>
#include<algorithm>
#include<chrono>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
using std::cout;
using std::default_random_engine;
using std::uniform_real_distribution;
using std::endl;
using std::vector;
using std::sort;
using std::chrono::steady_clock;
using std::chrono::duration;

vector<double> *InitializeVec(int, int);
double *Initialize(int, int);
void check(vector<double> *);
struct KeysLogits{
    int sizeKeys;
    int sizeLogits;
    double *keys;
    double * logits;
};
void initializeMatrix(double * );
void initializeWeightsAndBias(double * , double * , double * , double * , double * , double *);
void hello();
#endif
