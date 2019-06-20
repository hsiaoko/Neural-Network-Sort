#include"graph.h"
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

	steady_clock::time_point predictStart = steady_clock::now();

	MatrixXf dense_1_kernel_0(1, 4);
	dense_1_kernel_0 <<39.29980451, -38.82846923, -38.37837488, 39.35526869;


	MatrixXf output_1 = (*x_in) * dense_1_kernel_0;
	MatrixXf dense_2_kernel_0(4, 1);

    dense_2_kernel_0<<37.80961724, -39.0512867, -39.18649525, 37.9512024;
	MatrixXf logits = output_1 * dense_2_kernel_0;

	(*keys_logits) << (*x_in), logits;

	steady_clock::time_point predictEnd = steady_clock::now();
	duration<double, std::milli> *timePredicte = new duration<double, std::milli>(predictEnd -predictStart);

	cout << "consumming of predict:" << timePredicte->count() << " ms" << endl;
	return 0;
}


