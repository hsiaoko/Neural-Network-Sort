#include"util.h"

/*vector<float> *Initialize(int size, int seed)
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
*/
float *Initialize(int size, int seed)
{
    default_random_engine e;
    uniform_real_distribution<float> distribution(0.0,10000.0);
    e.seed(seed);
    float *data = (float*)malloc(sizeof(float) * size);

    for (int i = 0; i < size; ++i){
        *(data + i) = distribution(e);

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
int max_(KeysLogits * keys_logits){
    float max_ = FLT_MIN;
    for (int i = 0; i < keys_logits->sizeKeys; ++i)
        if (*(keys_logits->logits + i) > max_){
            continue;
        }else{
            max_ = *(keys_logits->logits + i);
        }
    return max_;
}
