
#ifndef NEURON_INCLUDED
#define NEURON_INCLUDED

#include "stdafx.h"
#include <vector>
#include <random>

class Neuron
{

public:

	//constructor
	Neuron(); 
	~Neuron();

	//methods
	void set_inputs(std::vector<double>& inputs);         // set m_inputs vector
	void neurinit(std::default_random_engine& generator); // initialize weights randomly
	void compute();                                       // compute m_output given m_inputs
	void set_weights(std::vector<double>& weights);       // set the weights m_weights of the inputs
	std::vector<double> get_weights();                    // get m_weights
	double get_output();                                  // get m_output

private:

	std::vector<double> m_inputs;    // vector of inputs of the neuron
	std::vector<double> m_weights;   // vector of weights applied to the inputs
	double m_output;                 // output of the neuron
};




#endif