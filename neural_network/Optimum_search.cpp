#include "stdafx.h"
#include <vector>
#include "Optimum_search.h"
#include "Neural_Network.h"
#include "Back_prop.h"

using namespace std;

Optimum_search::Optimum_search()
{
}

Optimum_search::~Optimum_search()
{
}

// initiale the back propagation parameters
void Optimum_search::init(double lambda, double alpha, double stop_crit, string save_path)
{
	m_lambda    = lambda;
	m_alpha     = alpha;
	m_stop_crit = stop_crit;
	m_save_path = save_path;
	m_cv_cost = {};
}

// set the training set
void Optimum_search::training_set(vector<vector<double>>& training_inputs, vector<vector<double>>& training_outputs)
{
	m_training_in  = training_inputs;
	m_training_out = training_outputs;
}


// set the cross validation set
void Optimum_search::cv_set(vector<vector<double>>& cv_inputs, vector<vector<double>>& cv_outputs)
{
	m_cv_in = cv_inputs;
	m_cv_out = cv_outputs;
}

// training of multiple networks
void Optimum_search::search(int N_input, int N_neuron_max, int N_layer_max, int N_output)
{
	vector<int> scheme_trunc = { N_input }; //truncated scheme vector
	vector<int> scheme;                     // full scheme vector
	Network network;
	vector<double> net_outputs;


	for (int i = 1; i <= N_layer_max; ++i)
	{
		scheme_trunc.push_back( 0 );

		for (int j = 0; j < N_neuron_max; ++j)
		{
			++scheme_trunc[i];

			scheme = scheme_trunc;
			scheme.push_back(N_output);

			network.build_network(scheme);

			Back_prop back_prop(network, m_lambda);
			back_prop.training(m_training_in, m_training_out, m_alpha, m_stop_crit, m_save_path);

			Back_prop::init();
			for (size_t k = 0 ; k < m_cv_in.size() ; ++k)
			{
				network.set_inputs(m_cv_in[k]);
				network.forward_prop();

				net_outputs = network.get_outputs();
				cost_sum(net_outputs, m_cv_out[k] );
			}

			cost();
			m_cv_cost.push_back(get_cost());
		}
	}




}