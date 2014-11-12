#include "stdafx.h"
#include <iostream>
#include <random>
#include "F:/Projets-C++/neural_network/neural_network/Layer.h"
#include <vector>
#include "F:/Projets-C++/neural_network/neural_network/Neuron.h"


using namespace std;


// Constructor
Layer::Layer(const int& Ninputs, const int& Nneurons, const int& layer_id) : m_Ninputs(Ninputs), m_Nneurons(Nneurons), m_layer_id(layer_id)
{
}

// destructor
Layer::~Layer()
{
}


// building layer
void Layer::build_layer(default_random_engine& generator)
{

	// temporary input vector, dimension m_Ninputs
	m_inputs.assign(m_Ninputs, 1.);

	//initialization of the output vector
	m_outputs.assign(m_Nneurons+1, 1.); // with the bias unit


	for (int i = 0; i < m_Nneurons; ++i)
	{
		m_layer[i].set_inputs(m_inputs);
		m_layer[i].neurinit(generator);
	}

	unrolled_weights();

}


// unrolled weights in the layer
void Layer::unrolled_weights()
{
	vector<double> weights;

	for (int i = 0; i < m_Nneurons; ++i)
	{

		weights = m_layer[i].get_weights();

		for (size_t j = 0; j < weights.size(); ++j)
		{
			m_allWeights.push_back(weights[j]);
		}

	}
}




// Set all coefficients to value coeff
void Layer::set_allWeights(vector<double>& weights)
{


	if (weights.size() == m_allWeights.size())
	{

		m_allWeights = weights;

		int N = weights.size()/m_Nneurons;
		vector<double> nweights;

		for (int i = 0; i < m_Nneurons; ++i)
		{

			nweights.clear();

			for (int j = 0; j < N; ++j)
			{
				nweights.push_back(weights[i*N + j]);
			}

			m_layer[i].set_weights( nweights);
		}
	
	}
	else
	{
		cout << "ERROR: wrong vector size for unrolled weights in Layer " << m_layer_id << endl;
		cout << "m_allWeights size  = " << m_allWeights.size() << endl;
		cout << "input weights size = " << weights.size() << endl;
		exit(1);
	}
	
}


// seting input vector
void Layer::set_inputs(vector<double>& inputs)
{
	m_inputs = inputs;

	for (int i = 0; i < m_Nneurons; ++i)
	{
		m_layer[i].set_inputs(m_inputs);
	}
}

// computing outputs
void Layer::compute()
{
	for (int i = 0; i < m_Nneurons; ++i)
	{
		m_layer[i].compute();
		m_outputs[i] = m_layer[i].get_output();
	}

	//m_outputs.push_back(1.); // adding the bias unit
}

// print output of all neurons in layer
void Layer::print_outputs()
{

	cout << " layer_id out = " << m_layer_id << ": ";

	for (auto x : m_outputs)
	{
		cout << x << " ";
	}

	cout << endl;
}


// print output of all neurons in layer
void Layer::print_inputs()
{

	cout << " layer_id in  = " << m_layer_id << ": ";

	for (auto x : m_inputs)
	{
		cout << x << " ";
	}

	cout << endl;
}

// returning the output vector
vector<double> Layer::get_outputs()
{
	return m_outputs;
}


// getting all neuron weights in the layer
vector<double> Layer::get_weights()
{
	return m_allWeights;
}