
#include <fstream>
#include <iostream>
#include "Back_Prop.h"
#include "Neural_Network.h"
#include "Grad_check.h"


using namespace std;


namespace NeuralNetwork
{

	//step of gradient computing with finite difference method
	void Grad_check::grad_step(vector<double>& training_inputs, vector<double>& training_outputs)
	{
		vector<vector<double>> weights_min;
		vector<vector<double>> weights_max;

		Network network_min = m_network;
		Network network_max = m_network;

		double cost_min;
		double cost_max;

		for (size_t i = 0; i < m_net_weights.size(); ++i)
		{

			for (size_t j = 0; j<m_net_weights[i].size(); ++j)
			{
				weights_min[i][j] = m_net_weights[i][j] - m_epsilon;
				weights_max[i][j] = m_net_weights[i][j] + m_epsilon;
			}
		}

		network_min.set_allWeights(weights_min);
		network_min.forward_prop();

		vector<double> net_outputs_min = network_min.get_outputs();

		m_cost_vect = {};


	}



	// Automatic training of the network
	void Grad_check::training(vector<vector<double>>& training_inputs, vector<vector<double>>& training_outputs, double alpha, double stop_crit,
		double epsilon, string path)
	{

		m_cost_vect = {};
		m_epsilon = epsilon;

		// perform training
		do
		{
			// initialize training
			init();

			for (size_t i = 0; i < training_inputs.size(); ++i)
			{
				cout << endl;
				cout << " training : " << i << endl;
				grad_step(training_inputs[i], training_outputs[i]);

				vector<double> net_outputs = m_network.get_outputs();
				cost_sum(net_outputs, training_outputs[i]);
			}

			cost();
			m_cost_vect.push_back(m_cost);

			// printing cost_vect
			ofstream file(path.c_str());

			if (file)
			{
				for (size_t j = 0; j < m_cost_vect.size(); ++j)
				{
					file << j << ", " << m_cost_vect[j] << endl;
				}
			}
			else
			{
				cout << "ERROR in Back_prop : can't open file in " << path.c_str() << endl;
			}

			// compute gradients from back prop algorithm
			back_prop_grads();

			// modify network's weights using gradient descent
			gradient_descent(alpha);


		} while (m_cost >= stop_crit);



	}



}
