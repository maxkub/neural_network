#ifndef INCLUDED_BACK_PROP
#define INCLUDED_BACK_PROP

#include "stdafx.h"
#include <vector>
#include "Neural_Network.h"

class Back_prop
{

public:

	// constructor
	Back_prop(Network& network, double& lambda);
	~Back_prop();


	void init();
	void back_propagation_step(std::vector<double>& training_inputs, std::vector<double>& training_outputs); // apply back prop algorithm to the training data : to compute m_Deltas
	void back_prop_grads();                      // compute gradients using the results of back_propagation, using m_Deltas
	void gradient_check();                       // check the gradients using finite difference method
	void gradient_descent(double alpha);         // perform gradient descent using m_Dvect
	void cost(std::vector<double>& training_outputs); // compute cost function : to use once the whole training set has been used

	void training(std::vector<std::vector<double>>& training_inputs, std::vector<std::vector<double>> training_outputs, double alpha, double stop_crit,
		std::string path);  // automatic training of the network

	void save(std::string& path);   // save current state of training (for large training data sets)
	void load(std::string& path);   // loading the last saved state in training
	
	double get_cost();
	std::vector<double> get_cost_vect();

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

#endif