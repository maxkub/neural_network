#ifndef INCLUDED_BACK_PROP
#define INCLUDED_BACK_PROP

#include <vector>
#include "Neural_Network.h"


namespace NeuralNetwork
{

	class Back_prop
	{


	public:

		// constructor
		__declspec(dllexport) Back_prop(); // default constructor
		__declspec(dllexport) Back_prop(Network& network, double& lambda);
		__declspec(dllexport) ~Back_prop();


		__declspec(dllexport) void init();

		// apply back prop algorithm to the training data : to compute m_Deltas
		__declspec(dllexport) void back_propagation_step(std::vector<double>& training_inputs, std::vector<double>& training_outputs);

		// compute gradients using the results of back_propagation, using m_Deltas
		__declspec(dllexport) void back_prop_grads();

		// perform gradient descent using m_Dvect
		__declspec(dllexport) void gradient_descent(double alpha);


		// automatic training of the network
		__declspec(dllexport) void training(std::vector<std::vector<double>>& training_inputs, std::vector<std::vector<double>>& training_outputs, double alpha, double stop_crit,
			bool print=true);

		// printing the results of training
		__declspec(dllexport) void prints();

		// save current state of training (for large training data sets)
		__declspec(dllexport) void save(std::string& path);

		// loading the last saved state in training
		__declspec(dllexport) void load(std::string& path);

		//__declspec(dllexport) double get_cost();

		__declspec(dllexport) std::vector<double> get_cost_vect();

		__declspec(dllexport) std::vector<std::vector<double>> get_mod_weights();

		__declspec(dllexport) void set_save_path(std::string& path);

	private:

		Network m_network;                              // network to be trained
		std::vector<int> m_scheme;                      // scheme of the network
		std::vector<std::vector<double>> m_net_weights; // the network weights

		int m_training_num;                             // number of training examples (is updated after each successfull back_propagation)       
		std::vector<double> m_cost_vect;
		double m_lambda;                                // regularization parameter
		std::vector<std::vector<double>> m_Deltas;      // array of Deltas for back prop (size = (Neurons * Ninputs : unrolled) * Nlayer)
		std::vector<std::vector<double>> m_Dvect;       // vectors (for all layer) of the derivatives computed with back prop algorithm (size = (Neurons * Ninputs : unrolled) * Nlayer)
		std::vector<std::vector<double>> m_grads;       // vectors (for all layer) of the gradients computed with finite difference (size = (Neurons * Ninputs : unrolled) * Nlayer)

		std::string m_save_path;                        // path to save results
	};


}


#endif