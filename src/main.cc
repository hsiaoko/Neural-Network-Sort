#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include"cuda.cuh"
#include "quick.h"
#include"graph.h"
#include"util.h"
#include"map.h"
using namespace std;
using namespace Eigen;

void CpuModel()
{
    //initialize
	int size = 60000000;
	//60000000
	MatrixXd keys_logits(size, 2);
	vector<float> *data = Initialize(size, 2);
	cout << "data_size:" << data->size() << endl;
	MatrixXd x_in = ArrayXXd::Zero(data->size(), 1);

	for (int i = 0; i < data->size(); ++i)
	{
		x_in(i) = (*data)[i];
	}
	InitResut initResult;
	vector<double> *finalResult = new vector<double>();

    //Prediction 
    cout<<"Prediction start:\n"<<endl;
	steady_clock::time_point sortStart = steady_clock::now();
	steady_clock::time_point predictStart = steady_clock::now();
	graph_(&x_in, size, &keys_logits);
	steady_clock::time_point predictEnd = steady_clock::now();
    cout<<"Prediction end\n"<<endl;

	duration<double, std::milli> *timePredicte = new duration<double, std::milli>(predictEnd -predictStart);
	auto max = keys_logits.colwise().maxCoeff();
    
    //To bucket
	initResult.sortedList = new vector<double>(int(round(max[1])*3), FLT_MAX);
	initResult.waitedList = new vector<double>();
    cout<<"To bucket start:\n"<<endl;
	steady_clock::time_point sortBucketStart = steady_clock::now();
	ToBucket(&keys_logits, &initResult);
	steady_clock::time_point sortBucketEnd = steady_clock::now();
    cout<<"To bucket end\n"<<endl;


    // Quick sort & merge
    cout<<"Quick sort & merge start:\n"<<endl;

	steady_clock::time_point quicksortStart = steady_clock::now();
	QuickSort(initResult.waitedList, 0, initResult.waitedList->size() - 1);
	steady_clock::time_point quicksortEnd = steady_clock::now();

	steady_clock::time_point mergeStart = steady_clock::now();
	Merge(&initResult, finalResult);
	steady_clock::time_point mergeEnd = steady_clock::now();
	steady_clock::time_point sortEnd = steady_clock::now();
    cout<<"Quick sort & merge start end\n"<<endl;

    //Statistic & check
    cout<<"Statistic & check:\n"<<endl;
	duration<double, std::milli> *timeSpanBucket = new duration<double, std::milli>(sortBucketEnd - sortBucketStart);
	duration<double, std::milli> *timeSpanQuicksort = new duration<double, std::milli>(quicksortEnd - quicksortStart);
	duration<double, std::milli> *timeSpanMerg = new duration<double, std::milli>(mergeEnd - mergeStart);
	duration<double, std::milli> *timeSpan1 = new duration<double, std::milli>(sortEnd - sortStart);

	cout << "consumming of predict:" << timePredicte->count() << " ms" << endl;
	cout << "consumming of bucket:" << timeSpanBucket->count() << " ms" << endl;
	cout << "consumming of quicksort:" << timeSpanQuicksort->count() << " ms" << endl;
	cout << "consumming of merge:" << timeSpanMerg->count() << " ms" << endl;
	cout << "consumming of sort:" << timeSpan1->count() << " ms" << endl;

    check(finalResult);
	// cout << "--------------------finalReuslt---------------------"<< endl;
	// for(int i=0;i<finalResult->size();++i)
	// {
	// 	cout<<(*finalResult)[i]<<endl;
	// }
	//	graph_();
    cout<<"Sort finish";
}

int main(){
    model();

}

