#ifndef TOOL_H
#define TOOL_H
#include <iostream>
#include <vector>
#include <random>
#include<algorithm>
#include<chrono>
#include<float.h>
using std::cout;
using std::default_random_engine;
using std::uniform_real_distribution;
using std::endl;
using std::vector;
using std::sort;
using std::chrono::steady_clock;
using std::chrono::duration;

//vector<float> *Initialize(int, int);
float *Initialize(int, int);
void check(vector<double> *);
struct KeysLogits{
    int sizeKeys;
    int sizeLogits;
    float *keys;
    float * logits;
};
int max_(KeysLogits * );
#endif
