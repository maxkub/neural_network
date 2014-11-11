#include "stdafx.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "Neural_Network.h"

using namespace std;

int main()
{

	//int Ninputs = 5, Noutputs = 2, Nrows = 6, Nlayers = 3;
	double coef = 1.;
	vector<int> scheme = { 3, 2, 1 };
	vector<double> inputs = {1.,1.,1.};
	vector<double> outputs;
	vector<vector<double>> weights;


	// create random number generator
	default_random_engine generator;

	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	generator.seed(seed);


	//Network network(Ninputs, Noutputs, Nrows, Nlayers, 1);
	Network network(scheme, 1);

	network.build_network(generator);

	weights = network.get_allWeights();

	cout << "all weights size = " << weights.size() << endl;

	for (auto c : weights)
	{
		for (auto x : c)
		{
			cout << x << " ";
		}
		cout << endl;
		
	}
	cout << endl;
	

	// assign 1 to all weights
	/*

	for (int i = 0; i < scheme.size() - 1; ++i)
	{
		weights[i].assign( (scheme[i]+1) * scheme[i + 1], coef);
	}

	network.set_allWeights(weights);
	*/

	// setting inputs
	network.set_inputs(inputs);

    // forward prop
	network.forward_prop();

	outputs = network.get_outputs();


	cout << "Outputs : \n";
	for (int i = 0; i < outputs.size(); ++i)
	{
		cout << outputs[i] << endl;
	}


	return 0;
}

