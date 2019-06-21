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
MatrixXd *graph_(MatrixXd *x_in, int data_size, MatrixXd *keys_logits)
{


	MatrixXd dense_1_kernel_0(1, 4);
	VectorXd dense_1_bias_0(4);
	MatrixXd dense_2_kernel_0(4, 1);
    VectorXd logits_bias(1);



	dense_1_kernel_0 <<39.29980451, -38.82846923, -38.37837488,39.35526869;
	dense_1_bias_0 << 26.85578852, -26.42188599, -26.51935639, 27.0847777;
    dense_2_kernel_0 << 37.80961724, -39.0512867, -39.18649525, 37.9512024;
    logits_bias<< 23.55981952;
    
	MatrixXd output_1 = (*x_in) * dense_1_kernel_0;

    output_1.rowwise() += dense_1_bias_0.transpose();



	MatrixXd logits = output_1 * dense_2_kernel_0;
    logits.rowwise() += logits_bias.transpose();

	(*keys_logits) << (*x_in), logits;


	return 0;
}


