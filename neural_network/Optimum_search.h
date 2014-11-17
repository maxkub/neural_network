#include "stdafx.h"
#include "Back_prop.h"
#include <string>

class Optimum_search : public Back_prop
{
public:

	Optimum_search();
	~Optimum_search();

	void init(double lambda, double alpha, double stop_crit, std::string save_path);

	void training_set(std::vector<std::vector<double>>& training_inputs, std::vector<std::vector<double>>& training_outputs); // seting the training set

	void cv_set(std::vector<std::vector<double>>& cv_inputs, std::vector<std::vector<double>>& cv_outputs);       // setting cross validation set

	void search(int N_input, int N_neuron_max, int N_layer_max, int N_output);      // training of multiple networks

private:

	std::vector<std::vector<double>> m_training_in;   // training set inputs
	std::vector<std::vector<double>> m_training_out;  // training set outputs
	std::vector<std::vector<double>> m_cv_in;         // cross validation set inputs
	std::vector<std::vector<double>> m_cv_out;        // cross validation set outputs
	double m_lambda;                                  // regularization parameter 
	double m_alpha;                                   // gradient descent parameter
	double m_stop_crit;                               // stop criterium for training
	std::string m_save_path;                          // path to save the outputs
	std::vector<double> m_cv_cost;                         // to compute the cost on cross validation set



};