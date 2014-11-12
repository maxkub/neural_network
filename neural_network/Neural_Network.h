#ifndef HLAYERS_INCLUDED
#define HLAYERS_INCLUDED

#include "stdafx.h"
#include <vector>
#include "F:/Projets-C++/neural_network/neural_network/Neuron.h"
#include "F:/Projets-C++/neural_network/neural_network/Layer.h"

class Network
{

public:

	//constructor
	Network(std::vector<int>& scheme, int print = 0);
	~Network();

	//methods
	void build_network(int seed=0);
	void set_allWeights(std::vector<std::vector<double>>& weights);
	void set_inputs(std::vector<double>& inputs);
	void forward_prop();

	std::vector<double> get_outputs();
	std::vector<std::vector<double>> get_allWeights();

	//double get_input_size();
	//double get_Noutputs();

	std::vector<double> get_layer_outputs(int& num);
	std::vector<int> get_scheme();

	void save(std::string path); // saving the network (scheme and weights)
	void load(std::string path); // loading a network (scheme and weights)

private:

	void unrolled_weights(); // build a vector of unrolled weights m_allweights

	int m_prints;           // 0: No printing , 1: print output of all neurons
	int m_Nlayers;          // Number of hidden layers
	int m_input_size;       // Lenght of the input vector (without bias)
	int m_Noutputs;         // Number of neurons in the output layer 
	std::vector<double> m_inputs;             // Input vector (size = m_input_size)
	std::vector<double> m_outputs;            // Output vector (size = m_Noutputs)
	std::vector<std::vector<double>> m_allweights; // all neuron weights (layers weights are unrolled) in the network (size = (m_input_size+1)*m_Nrows + m_Nlayers*(m_Nrows+1) + m_Noutputs*(m_Nrows+1) ) +1 is for bias
	std::vector<Layer> m_network;             // vector of all layers (size = m_Nlayers+1) +1 is for the output layer

	std::vector<int> m_scheme; // scheme of the network = {m_input_size, m_Nrows, ..., m_Nrows, m_Noutputs} without the bias units, m_scheme.size() = total layers
};



#endif