#ifndef INCLUDED_OPTSEARCH
#define INCLUDED_OPTSEARCH


#include "Back_Prop.h"
#include <string>

namespace NeuralNetwork
{

	class Optimum_search 
	{
	public:

		__declspec(dllexport) Optimum_search();
		__declspec(dllexport) ~Optimum_search();

		__declspec(dllexport) void init(double& lambda, double& alpha, double& stop_crit);

		// seting the training set
		__declspec(dllexport) void training_set(std::vector<std::vector<double>>& training_inputs, std::vector<std::vector<double>>& training_outputs);

		// setting cross validation set
		__declspec(dllexport) void cv_set(std::vector<std::vector<double>>& cv_inputs, std::vector<std::vector<double>>& cv_outputs);

		// training of multiple networks
		__declspec(dllexport) void search(int N_input, int N_neuron_max, int N_layer_max, int N_output, bool print=true);

		// setting the save path
		__declspec(dllexport) void set_save_path(std::string& save_path);

		// printing the results
		__declspec(dllexport) void prints();

	private:

		std::vector<std::vector<double>> m_training_in;   // training set inputs
		std::vector<std::vector<double>> m_training_out;  // training set outputs
		std::vector<std::vector<double>> m_cv_in;         // cross validation set inputs
		std::vector<std::vector<double>> m_cv_out;        // cross validation set outputs
		double m_lambda;                                  // regularization parameter 
		double m_alpha;                                   // gradient descent parameter
		double m_stop_crit;                               // stop criterium for training
		std::string m_save_path;                          // path to save the outputs
		std::vector<double> m_training_cost;              // costs for training set for tested networks
		std::vector<double> m_cv_cost;                    // costs for cross validation set for tested networks
		std::vector<int> m_N_neurons_tot;                 // total number of neurons in successive tested networks



	};

}


#endif