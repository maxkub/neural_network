#include "stdafx.h"
#include "Neuron.h"
#include <iostream>
#include <vector>
#include <random>

using namespace std;


// default constructor
Neuron::Neuron()
{
}

//constructor

/*Neuron::Neuron(vector<double>& inputs) : m_inputs(inputs)
{ 
}
*/

//destructor
Neuron::~Neuron()
{
}

// Setting input vector
void Neuron::set_inputs(vector<double>& inputs)
{
	m_inputs = inputs;
}

//initialize weights
void Neuron::neurinit(default_random_engine& generator)
{

	uniform_real_distribution<double> distribution(-5, 5);

	//initialize with random weights
	for (size_t i=0; i < m_inputs.size(); ++i)
	{
		m_weights.push_back(distribution(generator));
	}

	cout << "input size = " << m_weights.size() << endl;
}


// setting neuron weights (for backward propagation)
void Neuron::set_weights(vector<double>& weights)
{
	m_weights = weights;
}


// getting neuron weights
vector<double> Neuron::get_weights()
{
	return m_weights;
}


// neuron output computing
void Neuron::compute()
{
	double H=0.;

	for (size_t i=0; i < m_weights.size(); ++i)
	{
		H += m_inputs[i] * m_weights[i];
	}

	// Using sigmoid
	if (H <= -20.) // to prevent overflow problems
	{
		m_output = 0.;
	}
	else if (H >= 20.) // to prevent overflow problems
	{
		m_output = 1.;
	}
	else 
	{
		m_output = 1. / (1 + exp(-H));
	}

	//m_output = 1. / (1 + exp(-H));
	
}


// get neuron output
double Neuron::get_output()
{
	return m_output;
}