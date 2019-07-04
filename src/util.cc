#include"util.h"
using namespace std;

vector<double> *InitializeVec(int size, int seed)
{
    default_random_engine e;
    uniform_real_distribution<double> distribution(0.0,100000.0);
    e.seed(seed);
    vector<double> *data = new vector<double>(size);
    for (vector<double>::iterator iter = data->begin(); iter != data->end(); iter++)
    {
        *iter = distribution(e);
    }
    return data;
}

double *Initialize(int size, int seed)
{
    default_random_engine e;
    uniform_real_distribution<double> distribution(0.0,10000000.0);
    e.seed(seed);
    double *data = (double*)malloc(sizeof(double) * size);

    for (int i = 0; i < size; ++i){
        *(data + i) = distribution(e);

    }
    return data;
}
void initializeMatrix(double * matrix){

    for (int i =0 ; i<4;++i){
        for (int j = 0; j<8; ++j){
            *(matrix+i*8+j) = i;
        }
    }

    return;
}
void check(vector<double> *finalResul){
    //double minimum_ = FLT_MIN;
    double minimum_ = -1;
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
void initializeWeightsAndBias(double * weights_1_8, double * bias_1_8, double * weights_8_4, double * bias_8_4, double * weights_4_1, double *bias_4_1){

	FILE * pFile;

	//int matrixX = 1;
	//int matrixy = 8;
	//buffer = (double*)malloc(sizeof(double)*matrixX*matrixy);


	if((pFile = fopen ("/home/special/user/local/zxk/nns/Neural-Network-Sort-debug/params/1-1M_uniform_67128864_weight/weights_1_8.b","r")) == NULL){
		cout<<"can't open weights_1_8.txt"<<endl;
	}else{
		fread(weights_1_8, sizeof(double),1*8, pFile);
		fclose(pFile);
	}
	if((pFile = fopen ("/home/special/user/local/zxk/nns/Neural-Network-Sort-debug/params/1-1M_uniform_67128864_weight/weights_8_4.b","r")) == NULL){
		cout<<"can't open weights_8_4.txt"<<endl;
	}else{
		fread(weights_8_4, sizeof(double),8*4, pFile);
		fclose(pFile);
	}
	if((pFile = fopen ("/home/special/user/local/zxk/nns/Neural-Network-Sort-debug/params/1-1M_uniform_67128864_weight/weights_4_1.b","r")) == NULL){
		cout<<"can't open weight_4_1.txt"<<endl;
	}else{
		fread(weights_4_1, sizeof(double),4*1, pFile);
		fclose(pFile);
	}
	if((pFile = fopen ("/home/special/user/local/zxk/nns/Neural-Network-Sort-debug/params/1-1M_uniform_67128864_weight/bias_1_8.b","r")) == NULL){
		cout<<"can't open bias_1_8.txt"<<endl;
	}else{
		fread(bias_1_8, sizeof(double),1*8, pFile);
		fclose(pFile);
	}
	if((pFile = fopen ("/home/special/user/local/zxk/nns/Neural-Network-Sort-debug/params/1-1M_uniform_67128864_weight/bias_8_4.b","r")) == NULL){
		cout<<"can't open bias_8_4.txt"<<endl;
	}else{
		fread(bias_8_4, sizeof(double), 8*4, pFile);
		fclose(pFile);
	}
	if((pFile = fopen ("/home/special/user/local/zxk/nns/Neural-Network-Sort-debug/params/1-1M_uniform_67128864_weight/bias_4_1.b","r")) == NULL){
		cout<<"can't open bias_4_1.txt"<<endl;
	}else{
		fread(bias_4_1, sizeof(double),4*1, pFile);
		fclose(pFile);
	}
}
void hello(){
    cout<<"hello world"<<endl;
}
