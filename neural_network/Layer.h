#ifndef INCLUDED_LAYER
#define INCLUDED_LAYER

#include <vector>
#include "Neuron.h"

namespace NeuralNetwork
{

	class Layer

	{

	public:

		//constructor
		__declspec(dllexport) Layer(const int& Ninputs, const int& Nneurons, const int& layer_id);
		__declspec(dllexport) ~Layer();

		__declspec(dllexport) void build_layer(std::default_random_engine& generator);  // build the layer with m_Ninputs and m_Nneurons, initialze all weights randomly and build m_allWeights
		__declspec(dllexport) void set_allWeights(std::vector<double>& weights);        // set value of all the weights in the layer: m_allWeights 
		__declspec(dllexport) void set_inputs(std::vector<double>& inputs);             // set m_inputs
		__declspec(dllexport) void compute();                                           // compute the outputs of the layer: m_outputs
		__declspec(dllexport) void print_outputs();                                     // print m_outputs on screen 
		__declspec(dllexport) void print_inputs();                                      // print m_inputs on screen
		__declspec(dllexport) std::vector<double> get_outputs();                        // get m_outputs
		__declspec(dllexport) std::vector<double> get_weights();                        // get m_allWeights


	private:

		void unrolled_weights();   // function to build m_allWeights vector

		const int m_layer_id;      // Layer id
		const int m_Ninputs;       // Number of inputs of the layer (can encompass bias)
		const int m_Nneurons;      // Number of neurons in layer 

		std::vector<double> m_allWeights;   // vector of unrolled weights of the layer (size = m_Ninputs * m_Nneurons)
		std::vector<double> m_inputs;       // vector of inputs (size = m_Ninputs)
		std::vector<double> m_outputs;      // vector of outputs of the layer (size = m_Nneurons + 1 bias unit)

		Neuron * m_layer = new Neuron[m_Nneurons]; // vector of neurons = the layer (size = m_Nneurons)
	};


}





#endif