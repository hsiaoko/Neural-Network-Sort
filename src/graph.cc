#include"graph.h"
//using namespace ArrayXXf;
MatrixXf relu_(MatrixXf input)
{
	//cout<<input(0,0);
	MatrixXf *output;
	for (int i = 0; i < input.size(); ++i)
	{
		if (input(i) > 0)
		{
			continue;
		}
		else
		{
			input(i) = 0;
		}
	};
	return input;
}
MatrixXf max_out_(MatrixXf input, int neuron_number)
{

}
MatrixXf *graph_(MatrixXf *x_in, int data_size, MatrixXf *keys_logits)
{

	MatrixXf dense_1_kernel_0(1, 8);
	dense_1_kernel_0 << 0.16125606, 0.52015157, 0.4085207, 0.03481522, -0.27285749, 0.42178148, -0.50626113, -0.4243784;

	MatrixXf dense_2_kernel_0(8, 4);
	dense_2_kernel_0 << 0.39439846, 0.09576376, 0.33283754, -0.05390806,
		-0.03016981, 0.30637967, -0.65281011, -0.16486718,
		0.06982948, 0.32160448, 0.5196324, -0.66382376,
		0.28130285, 0.26371781, 0.021555, 0.5641126,
		0.4477407, 0.58044386, -0.25365363, 0.55271017,
		0.63562647, 0.31281738, 0.37089121, 0.17723571,
		-0.3695696, -0.69711928, 0.66131771, 0.23091989,
		-0.6346922, 0.28612702, -0.65594568, 0.19090521;

	MatrixXf dense_3_kernel_0(4, 1);
	dense_3_kernel_0 << 0.61818643, -0.2919698, -0.41026983, 0.00757869;



	steady_clock::time_point computeStart = steady_clock::now();

	MatrixXf output_1 = (*x_in) * dense_1_kernel_0;
	steady_clock::time_point relu1 = steady_clock::now();
	MatrixXf y_1 = relu_(output_1); 
	steady_clock::time_point relu2 = steady_clock::now();

	MatrixXf output_2=y_1 * dense_2_kernel_0;
	steady_clock::time_point relu3 = steady_clock::now();
	MatrixXf y_2 = relu_(output_2);
	steady_clock::time_point relu4 = steady_clock::now();

	MatrixXf output_3=y_2 * dense_3_kernel_0;
	steady_clock::time_point relu5 = steady_clock::now();
	MatrixXf logits = relu_(output_3);
	steady_clock::time_point relu6 = steady_clock::now();

	steady_clock::time_point computeEnd = steady_clock::now();
	duration<double, std::milli> *timeSpanRelu1 = new duration<double, std::milli>(relu2 - relu1);
	duration<double, std::milli> *timeSpanRelu2 = new duration<double, std::milli>(relu4 - relu3);
	duration<double, std::milli> *timeSpanRelu3 = new duration<double, std::milli>(relu6 - relu5);
	duration<double, std::milli> *timeSpanGraph = new duration<double, std::milli>(computeEnd - computeStart);
	cout << "consumming of relu:" << timeSpanRelu1->count()+timeSpanRelu2->count()+timeSpanRelu3->count() << " ms" << endl;
	cout << "consumming of graph:" << timeSpanGraph->count() << " ms" << endl;

	(*keys_logits) << (*x_in), logits;
	return 0;
}


