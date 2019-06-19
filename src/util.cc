#include"../include/util.h"

vector<float> *Initialize(int size, int seed)
{
    default_random_engine e;
    uniform_real_distribution<float> distribution(0.0,10000.0);
    e.seed(seed);
    vector<float> *data = new vector<float>(size);
    for (vector<float>::iterator iter = data->begin(); iter != data->end(); iter++)
    {
        *iter = distribution(e);
    }
    return data;
}