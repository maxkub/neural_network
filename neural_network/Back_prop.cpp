#include "stdafx.h"
#include <fstream>
#include <iostream>
#include "F:/Projets-C++/neural_network/neural_network/Back_prop.h"
#include "F:/Projets-C++/neural_network/neural_network/Neural_Network.h"
#include <math.h>

using namespace std;

// Constructor
Back_prop::Back_prop(Network& network, double& lambda) : m_network(network), m_lambda(lambda)
{
}

// Destructor
Back_prop::~Back_prop()
{
}


// initialization of the training
void Back_prop::init()
{
	m_training_num = 0;
	m_cost = 0.;
	m_Deltas = {};
	m_scheme = m_network.get_scheme();
	m_net_weights = m_network.get_allWeights();

	m_Deltas = m_net_weights;

	for (size_t i = 0; i < m_Deltas.size(); ++i)
	{
		m_Deltas[i].assign(m_Deltas[i].size(), 0.);
	}

	m_Dvect = m_Deltas;
	m_grads = m_Deltas;

	
}

// Backpropagation algorithm
void Back_prop::back_propagation_step(vector<double>& training_inputs, vector<double>& training_outputs)
{

	vector<double> net_outputs;
	vector<vector<double>> deltas;
	vector<double> layer_outputs;
	vector<double> temp_d;
	double sum=0.;


	m_network.set_inputs(training_inputs);
	m_network.forward_prop();

	net_outputs = m_network.get_outputs();

	// compute the error terms of the output layer
	temp_d.clear();
	for (size_t i = 0; i < net_outputs.size(); ++i)
	{
		//cout << "net_outputs.size = " << net_outputs.size() << endl;
		temp_d.push_back( (net_outputs[i] - training_outputs[i])*net_outputs[i]*(1.-net_outputs[i]) );
	}

	deltas.push_back(temp_d);


	// compute the error terms of the earlier layers
	int it = 0;
	for (size_t i = m_scheme.size() - 1; --i > 0;) // reverse loop on layers
	{

		temp_d.clear();

		//cout << endl;

		for (int j = 0; j <= m_scheme[i]; ++j) // loop on neurons in layer l (with the bias)
		{

			sum = 0.;

			for (int k = 0; k < m_scheme[i + 1]; ++k) // loop on neurons in layer l+1 (without the bias)
			{
				//cout << " ..." << i << " " << j << " " << k << " " << k*(m_scheme[i] + 1) + j << endl;
				sum += deltas[it][k] * m_net_weights[i][k*(m_scheme[i] + 1) + j];
			}

			int num = i - 1;
			layer_outputs = m_network.get_layer_outputs(num);

			temp_d.push_back(sum*layer_outputs[j] * (1. - layer_outputs[j]));


		}

		deltas.push_back(temp_d);
		++it;
	}


	//cout << "end deltas \n";

	// compute Deltas array
	for (size_t i = 0; i < m_net_weights.size(); ++i)
	{
		
		if (i == 0)
		{
			layer_outputs = training_inputs;
			layer_outputs.push_back(1.); // adding bias unit
		}
		else
		{
			int num = i - 1;
			layer_outputs = m_network.get_layer_outputs(num);
		}
		

		int it = 0;
		for (int j = 0; j < m_scheme[i+1]; ++j) // loop on neurons in layer l+1 (without bias unit)
		{
			for (int k = 0; k <= m_scheme[i]; ++k) // loop on neurons in layer l (with bias unit)
			{
				//cout << "m_deltas[i].size = " << m_Deltas[i].size() << endl;
				//cout << "layer_output.size = " << layer_outputs.size() << endl;

				//cout << "D.. " << i << " " << j << " " << k << " " << m_scheme.size() - 2 - i << endl;
				
				m_Deltas[i][it] += deltas[m_scheme.size() - 2 - i][j] * layer_outputs[k];

				++it;
			}
			
		}
		
	}

	m_training_num += 1;

	

}


// compute the gradients using back_propagation results stored in m_Deltas
void Back_prop::back_prop_grads()
{

	for (size_t i = 0; i < m_Dvect.size(); ++i)
	{

		for (size_t j = 0; j < m_Dvect[i].size(); ++j)
		{
			m_Dvect[i][j] = 1. / ((float)m_training_num)*m_Deltas[i][j]; //+ m_lambda*m_net_weights[i][j];

		}
	}
}

// gradient descent
void Back_prop::gradient_descent(double alpha)
{
	for (size_t i = 0; i < m_Dvect.size(); ++i)
	{

		for (size_t j = 0; j < m_Dvect[i].size(); ++j)
		{
			m_net_weights[i][j] -= alpha*m_Dvect[i][j];
		}
	}

	m_network.set_allWeights(m_net_weights);

}

// compute cost function
void Back_prop::cost(vector<double>& net_outputs, vector<double>& training_outputs)
{
	//vector<double> net_outputs = m_network.get_outputs();

	for (int i = 0; i < m_scheme.back(); ++i)
	{
		m_cost += -(training_outputs[i] * log(net_outputs[i]) + (1. - training_outputs[i]) * log(1. - net_outputs[i])) ;
	}
}



// Automatic training of the network
void Back_prop::training(vector<vector<double>>& training_inputs, vector<vector<double>>& training_outputs, double alpha, double stop_crit, string path)
{

	// init
	m_cost_vect = {};

    // perform training
	int it = 0;
	do 
	{

		cout << " training : " << it << endl;

		// re-initialize training
		init();

		// loop on the training set
		for (size_t i = 0; i < training_inputs.size(); ++i)
		{
			//cout << endl;
			//cout << " training : " << i << endl;

			back_propagation_step(training_inputs[i], training_outputs[i]);

			vector<double> net_outputs = m_network.get_outputs();
			cost(net_outputs, training_outputs[i]);
		}

		m_cost = m_cost / ((float)m_training_num);
		m_cost_vect.push_back(m_cost);

		// printing cost_vect
		ofstream file(path.c_str());

		if (file)
		{
			for (size_t j = 0; j < m_cost_vect.size(); ++j)
			{
				file << j << ", " << m_cost_vect[j] << endl;
			}
		}
		else
		{
			cout << "ERROR in Back_prop : can't open file in " << path.c_str() << endl;
		}

		// compute gradients from back prop algorithm
		back_prop_grads();

		// modify network's weights using gradient descent
		gradient_descent(alpha);

		++it;

	} while (m_cost >= stop_crit && it <= 100);
	

	
}




// get m_cost
double Back_prop::get_cost()
{
	return m_cost;
}

// get m_cost_vect
vector<double> Back_prop::get_cost_vect()
{
	return m_cost_vect;
}


// getting all weights
vector<vector<double>> Back_prop::get_mod_weights()
{
	return m_net_weights;
}