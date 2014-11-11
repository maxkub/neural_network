#ifndef INCLUDED_LAYER
#define INCLUDED_LAYER

#include "stdafx.h"
#include <vector>
#include "Neuron.h"


class Layer

{

public:
	//constructor
	Layer(const int& Ninputs, const int& Nneurons, const int& layer_id);
	~Layer();

	void build_layer(std::default_random_engine& generator);  // build the layer with m_Ninputs and m_Nneurons, initialze all weights randomly and build m_allWeights
	void set_allWeights(std::vector<double>& weights);        // set value of all the weights in the layer: m_allWeights 
	void set_inputs(std::vector<double>& inputs);             // set m_inputs
	void compute();                                           // compute the outputs of the layer: m_outputs
	void print_outputs();                                     // print m_outputs on screen 
	std::vector<double> get_outputs();                        // get m_outputs
	std::vector<double> get_weights();                        // get m_allWeights


private:

	void unrolled_weights();   // function to build m_allWeights vector

	const int m_layer_id;      // Layer id
	const int m_Ninputs;       // Number of inputs of the layer (can encompass bias)
	const int m_Nneurons;      // Number of neurons in layer 

	std::vector<double> m_allWeights;   // vector of unrolled weights of the layer (size = m_Ninputs * m_Nneurons)
	std::vector<double> m_inputs;       // vector of inputs (size = m_Ninputs)
	std::vector<double> m_outputs;      // vector of outputs of the layer (size = m_Nneurons)

	Neuron * m_layer = new Neuron[m_Nneurons]; // vector of neurons = the layer (size = m_Nneurons)
};





#endif