#ifndef INCLUDED_BACK_PROP
#define INCLUDED_BACK_PROP

#include <vector>
#include "Neural_Network.h"


namespace NeuralNetwork
{

	class __declspec(dllexport) Back_prop
	{


	public:

		// constructor
		Back_prop(); // default constructor
		Back_prop(Network& network, double& lambda);
		~Back_prop();


		void init();

		// apply back prop algorithm to the training data : to compute m_Deltas
		void back_propagation_step(std::vector<double>& training_inputs, std::vector<double>& training_outputs);

		// compute gradients using the results of back_propagation, using m_Deltas
		void back_prop_grads();

		// perform gradient descent using m_Dvect
		void gradient_descent(double alpha);

		// sum cost function terms
		void cost_sum(std::vector<double>& net_outputs, std::vector<double>& training_outputs);

		// compute cost function, with regularization terms
		void cost();

		// automatic training of the network
		void training(std::vector<std::vector<double>>& training_inputs, std::vector<std::vector<double>>& training_outputs, double alpha, double stop_crit,
			std::string path);

		// save current state of training (for large training data sets)
		void save(std::string& path);

		// loading the last saved state in training
		void load(std::string& path);

		double get_cost();

		std::vector<double> get_cost_vect();

		std::vector<std::vector<double>> get_mod_weights();

	private:

		Network m_network;         // network to be trained
		std::vector<int> m_scheme; // scheme of the network
		std::vector<std::vector<double>> m_net_weights; // the network weights

		int m_training_num;     // number of training examples (is updated after each successfull back_propagation)
		double m_cost;          // value of cost function on the training set
		std::vector<double> m_cost_vect;
		double m_lambda;        // regularization parameter
		std::vector<std::vector<double>> m_Deltas;  // array of Deltas for back prop (size = (Neurons * Ninputs : unrolled) * Nlayer)
		std::vector<std::vector<double>> m_Dvect;   // vectors (for all layer) of the derivatives computed with back prop algorithm (size = (Neurons * Ninputs : unrolled) * Nlayer)
		std::vector<std::vector<double>> m_grads;   // vectors (for all layer) of the gradients computed with finite difference (size = (Neurons * Ninputs : unrolled) * Nlayer)
	};


}


#endif