#include "stdafx.h"
#include <iostream>
#include <vector>
#include "F:/Projets-C++/neural_network/neural_network/Neural_Network.h"

using namespace std;

int main()
{

	double coef = 1.;
	vector<int> scheme = { 3, 2, 1 };
	vector<double> inputs = {1.,1.,1.};
	vector<double> outputs;
	vector<vector<double>> weights;



	Network network;

	network.build_network(scheme, 1 , 0);

	weights = network.get_allWeights();

	//cout << "all weights size = " << weights.size() << endl;

	/*
	for (auto c : weights)
	{
		for (auto x : c)
		{
			cout << x << " ";
		}
		cout << endl;
		
	}
	cout << endl;*/
	

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
	for (size_t i = 0; i < outputs.size(); ++i)
	{
		cout << outputs[i] << endl;
	}


	string path = "F:/Neural_networks_dat/weights_test.dat";
	network.save(path);

	Network network2;
	network2.import(path, 1);

	path = "F:/Neural_networks_dat/weights_test2.dat";
	network2.save(path);


	return 0;
}

