
#include <vector>
#include <fstream>
#include <iostream>
#include <limits>
#include "Optimum_search.h"
#include "Neural_Network.h"
#include "Back_Prop.h"

using namespace std;

namespace NeuralNetwork
{

	Optimum_search::Optimum_search()
	{
	}

	Optimum_search::~Optimum_search()
	{
	}

	// initiale the back propagation parameters
	void Optimum_search::init(double& lambda, double& alpha, double& stop_crit)
	{
		m_lambda = lambda;
		m_alpha = alpha;
		m_stop_crit = stop_crit;
		m_training_cost = {};
		m_cv_cost = {};
		m_N_neurons_tot = {};
	}

	// set the training set
	void Optimum_search::training_set(vector<vector<double>>& training_inputs, vector<vector<double>>& training_outputs)
	{
		m_training_in = training_inputs;
		m_training_out = training_outputs;
	}


	// set the cross validation set
	void Optimum_search::cv_set(vector<vector<double>>& cv_inputs, vector<vector<double>>& cv_outputs)
	{
		m_cv_in = cv_inputs;
		m_cv_out = cv_outputs;
	}

	// training of multiple networks
	void Optimum_search::search(int N_input, int N_neuron_max, int N_layer_max, int N_output, bool print)
	{
		vector<int> scheme_trunc = { N_input }; //truncated scheme vector
		vector<int> scheme;                     // full scheme vector
		vector<double> net_outputs;
		vector<double> cost_vect;

		int it = 0;


		for (int i = 1; i <= N_layer_max; ++i)
		{
			scheme_trunc.push_back(0);

			for (int j = 0; j < N_neuron_max; ++j)
			{
				++scheme_trunc[i];

				scheme = scheme_trunc;
				scheme.push_back(N_output);

				Network network;
				network.build_network(scheme);

				Back_prop back_prop(network, m_lambda);

				try
				{
					back_prop.training(m_training_in, m_training_out, m_alpha, m_stop_crit, false);

					cost_vect = back_prop.get_cost_vect();
					
					network.set_cost(0.);

					for (size_t k = 0; k < m_cv_in.size(); ++k)
					{
						network.set_inputs(m_cv_in[k]);
						network.forward_prop();

						//net_outputs = network.get_outputs();
						network.cost_sum(network.get_outputs(), m_cv_out[k]);
					}

					
					network.cost(static_cast<long>(m_cv_in.size()),m_lambda);

					cout << " cost " << network.get_cost() << endl;

					if (cost_vect.back() == numeric_limits<double>::infinity() || network.get_cost() == numeric_limits<double>::infinity())
					{
						m_training_cost.push_back(0.);
						m_cv_cost.push_back(0.);
					}
					else
					{
						m_training_cost.push_back(cost_vect.back());
						m_cv_cost.push_back(network.get_cost());
					}

					++it;
					
				}
				catch (...)
				{
					m_training_cost.push_back(0.);
					m_cv_cost.push_back(0.);
				}

				m_N_neurons_tot.push_back(it);

				prints();
				
				network.~Network();
				back_prop.~Back_prop();
				
			}
		}

	}

	// set m_save_path
	void Optimum_search::set_save_path(string& save_path)
	{
		m_save_path = save_path;
	}


	// printing the results
	void Optimum_search::prints()
	{
		ofstream file(m_save_path.c_str());

		if (file)
		{
			for (size_t i = 0; i < m_training_cost.size(); ++i)
			{
				file << m_N_neurons_tot[i] << " " << m_training_cost[i] << " " << m_cv_cost[i] << endl;
			}
		}
		else
		{
			cout << "ERROR in Optimum_search.print : can't open file " << m_save_path.c_str() << endl;
			exit(1);
		}
	}



}
