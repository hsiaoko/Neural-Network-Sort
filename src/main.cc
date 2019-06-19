#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "quick.h"
#include"graph.h"
#include"util.h"
#include"map.h"
using namespace std;
using namespace Eigen;
int main()
{
	int size = 2621440;
	//2621440
	MatrixXf keys_logits(size, 2);
	vector<float> *data = Initialize(size, 2);
	cout << "data_size:" << data->size() << endl;
	MatrixXf x_in = ArrayXXf::Zero(data->size(), 1);

	for (int i = 0; i < data->size(); ++i)
	{
		x_in(i) = (*data)[i];
	}
	steady_clock::time_point sortStart = steady_clock::now();
	graph_(&x_in, 3, &keys_logits);

	auto max = keys_logits.colwise().maxCoeff();
	InitResut initResult;
	initResult.sortedList = new vector<float>(int(round(max[1])), FLT_MAX);
	initResult.waitedList = new vector<float>();

	steady_clock::time_point sortBucketStart = steady_clock::now();
	ToBucket(keys_logits, &initResult);
	steady_clock::time_point sortBucketEnd = steady_clock::now();

	vector<float> *finalResult = new vector<float>();
	QuickSort(initResult.waitedList, 0, initResult.waitedList->size() - 1);
	steady_clock::time_point mergeStart = steady_clock::now();
	Merge(&initResult, finalResult);
	steady_clock::time_point mergeEnd = steady_clock::now();
	// cout << "--------------------finalReuslt---------------------"<< endl;
	// for(int i=0;i<finalResult->size();++i)
	// {
	// 	cout<<(*finalResult)[i]<<endl;
	// }
	steady_clock::time_point sortEnd = steady_clock::now();
	duration<double, std::milli> *timeSpanBucket = new duration<double, std::milli>(sortBucketEnd - sortBucketStart);
	duration<double, std::milli> *timeSpanMerg = new duration<double, std::milli>(mergeEnd - mergeStart);
	duration<double, std::milli> *timeSpan1 = new duration<double, std::milli>(sortEnd - sortStart);
	cout << "consumming of bucket:" << timeSpanBucket->count() << " ms" << endl;
	cout << "consumming of merge:" << timeSpanMerg->count() << " ms" << endl;
	cout << "consumming of sort:" << timeSpan1->count() << " ms" << endl;
	//	graph_();
}