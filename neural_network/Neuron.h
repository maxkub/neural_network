#ifndef NEURON_INCLUDED
#define NEURON_INCLUDED

#include <vector>
#include <random>

namespace NeuralNetwork
{

	class Neuron
	{

	public:

		//constructor
		__declspec(dllexport) Neuron();
		__declspec(dllexport) ~Neuron();

		//methods
		__declspec(dllexport) void set_inputs(std::vector<double>& inputs);         // set m_inputs vector
		__declspec(dllexport) void neurinit(std::default_random_engine& generator); // initialize weights randomly
		__declspec(dllexport) void compute();                                       // compute m_output given m_inputs
		__declspec(dllexport) void set_weights(std::vector<double>& weights);       // set the weights m_weights of the inputs
		__declspec(dllexport) std::vector<double> get_weights();                    // get m_weights
		__declspec(dllexport) double get_output();                                  // get m_output

	private:

		std::vector<double> m_inputs;    // vector of inputs of the neuron
		std::vector<double> m_weights;   // vector of weights applied to the inputs
		double m_output;                 // output of the neuron
	};

}



#endif