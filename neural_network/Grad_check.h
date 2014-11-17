#ifndef INCLUDED_GRAD_CHECK
#define INCLUDED_GRAD_CHECK

#include "stdafx.h"
#include <vector>
#include "Back_prop.h"

class Grad_check : public Back_prop
{

public:

	void grad_step(std::vector<double>& training_inputs, std::vector<double>& training_outputs);

	void training(std::vector<std::vector<double>>& training_inputs, std::vector<std::vector<double>>& training_outputs, double alpha, double stop_crit,
		double epsilon, std::string path);


private:

	double m_epsilon; // delta in finite difference methode

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