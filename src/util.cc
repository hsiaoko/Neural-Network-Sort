#include"util.h"

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
void check(vector<double> *finalResul){
    //float minimum_ = FLT_MIN;
    float minimum_ = -1;
    int count=0;
    vector<double>::iterator iter = finalResul->begin();
    for (; iter!=finalResul->end(); iter++){
     //   cout<<*iter<<endl;
        if (*iter >= minimum_){
            minimum_ = *iter;
            ++count;
        }else{
            break;
        }
    }
    if (iter == finalResul->end()){
        cout<<"order"<<endl;
    }else{
        cout<<"disorder "<<"because of "<<count<<" th-element"<<endl;
    }

}
