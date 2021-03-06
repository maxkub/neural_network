

#ifndef HLAYERS_INCLUDED
#define HLAYERS_INCLUDED


#include <vector>
#include "Neuron.h"
#include "Layer.h"

namespace NeuralNetwork
{

	class Network
	{

	public:

		//constructor
		__declspec(dllexport) Network();
		__declspec(dllexport) ~Network();

		//methods
		__declspec(dllexport) void build_network(std::vector<int>& scheme, int print = 0, int seed = 0);
		__declspec(dllexport) void set_allWeights(std::vector<std::vector<double>>& weights);
		__declspec(dllexport) void set_inputs(std::vector<double>& inputs);
		__declspec(dllexport) void forward_prop();

		__declspec(dllexport) std::vector<double> get_outputs();
		__declspec(dllexport) std::vector<std::vector<double>> get_allWeights();


		__declspec(dllexport) std::vector<double> get_layer_outputs(int& num);
		__declspec(dllexport) std::vector<int> get_scheme();

		// sum cost function terms
		__declspec(dllexport) void cost_sum(std::vector<double>& net_outputs, std::vector<double>& training_outputs);

		// compute cost function, with regularization terms
		__declspec(dllexport) void cost(int N_trainings, double lambda);

		__declspec(dllexport) void set_cost(double cost);

		__declspec(dllexport) double get_cost();

		__declspec(dllexport) void save(std::string path); // saving the network (scheme and weights)
		__declspec(dllexport) void import(std::string path, int print = 0); // loading a network (scheme and weights), and build it

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
		double m_cost; // to compute cost over a training or validation set
	};


}



#endif