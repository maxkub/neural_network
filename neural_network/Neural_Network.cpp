#include "stdafx.h"
#include "Neural_Network.h"
#include <vector>
#include <iostream>
#include <random>
#include "Neuron.h"
#include "Layer.h"


using namespace std;


// contructor
/*Network::Network(int& Ninputs, int& Noutputs, int& Nrows, int& Nlayers, int print) : m_input_size(Ninputs), m_Noutputs(Noutputs), m_Nrows(Nrows), m_Nlayers(Nlayers), m_prints(print)
{
}*/

Network::Network(vector<int>& scheme, int print) : m_scheme(scheme), m_input_size(scheme[0]), m_prints(print)
{
}

//destructor
Network::~Network()
{
}


// set network
void Network::build_network(default_random_engine& generator)
{

	m_input_size = m_scheme[0];
	m_Noutputs   = m_scheme.back();
	m_Nlayers    = m_scheme.size() - 2; // -Input -output



	for (size_t i = 0; i < m_scheme.size()-1; ++i)
	{
		Layer layer(m_scheme[i] + 1, m_scheme[i+1], i + 1);
		layer.build_layer(generator);

		m_network.push_back(layer);
	}


	unrolled_weights();

	
}

// unrolling layers weights
void Network::unrolled_weights()
{
	for (int i = 0; i <= m_Nlayers; ++i)
	{
		m_allweights.push_back(m_network[i].get_weights());
	}
}


// set inputs
void Network::set_inputs(vector<double>& inputs)
{
	if (inputs.size() == m_input_size) 
	{
		m_inputs = inputs;
		m_inputs.push_back(1.); // insert offset value in the vector's tail
	}
	else 
	{
		cout << "ERROR: wrong input size for network \n";
		cout << "m_inputs size = " << m_inputs.size() << endl;
		cout << "inputs size   = " << inputs.size() << endl;
	    exit(1);
	}
	

}

// Set the same value for all coeffcients
void Network::set_allWeights(vector<vector<double>>& weights)
{

	for (int i = 0; i <= m_Nlayers; ++i)
	{
		if (weights[i].size() == m_allweights[i].size())
		{
			m_allweights[i] = weights[i];
			m_network[i].set_allWeights(weights[i]);
		}
		else
		{
			cout << "ERROR : wrong size in network weights, layer id = " << i + 1 << endl;
			cout << "weights[id] size      = " << weights[i].size() << endl;
			cout << "m_allweights[id] size = " << m_allweights[i].size() << endl;
			exit(1);
		}
		
	}
}



// Forward propagation
void Network::forward_prop()
{

	vector<double> vect;

	m_network[0].set_inputs(m_inputs);
	m_network[0].compute();

	if (m_prints == 1)
	{
		m_network[0].print_outputs();
	}


	for (int i = 1; i <= m_Nlayers; ++i)
	{

		vect = m_network[i - 1].get_outputs();
		vect.push_back(1.); // adding offset value in vector's tail

		m_network[i].set_inputs(vect);
		m_network[i].compute();

		if (m_prints == 1)
		{
			m_network[i].print_outputs();
		}
	}

	m_outputs = m_network[m_Nlayers].get_outputs();
	

}


// Get outputs
vector<double> Network::get_outputs()
{
	return m_outputs;
}

// Get all weights in the network
vector<vector<double>> Network::get_allWeights()
{
	return m_allweights;
}


// Get m_input_size
/*double Network::get_input_size()
{
	return m_input_size;
}*/

// Get m_Noutputs
/*double Network::get_Noutputs()
{
	return m_Noutputs;
}*/

// Get outputs of layer id=num
vector<double> Network::get_layer_outputs(int& num)
{
	return m_network[num].get_outputs();
}


// Get network scheme
vector<int> Network::get_scheme()
{
	return m_scheme;
}
