
#include "Neural_Network.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include "Layer.h"


using namespace std;


namespace NeuralNetwork
{

	Network::Network()
	{
	}

	//destructor
	Network::~Network()
	{
	}


	// set network
	void Network::build_network(vector<int>& scheme, int print, int seed)
	{

		//initializations
		m_scheme = scheme;
		m_input_size = scheme[0];
		m_prints = print;
		m_cost = 0.;

		// create random number generator
		default_random_engine generator;

		if (seed == 0)
		{
			int lseed = chrono::system_clock::now().time_since_epoch().count();
			generator.seed(lseed);
		}
		else
		{
			generator.seed(seed);
		}


		m_input_size = m_scheme[0];
		m_Noutputs = m_scheme.back();
		m_Nlayers = m_scheme.size() - 2; // -Input -output



		for (size_t i = 1; i <= m_scheme.size() - 1; ++i)
		{
			Layer layer(m_scheme[i - 1] + 1, m_scheme[i], i);
			layer.build_layer(generator);

			m_network.push_back(layer);
		}

		cout << "end build" << endl;

		unrolled_weights();


	}

	// unrolling layers weights
	void Network::unrolled_weights()
	{
		for (size_t i = 0; i < m_scheme.size() - 1; ++i)
		{
			m_allweights.push_back(m_network[i].get_weights());
		}
	}


	// set inputs
	void Network::set_inputs(vector<double>& inputs)
	{
		if (inputs.size() == m_input_size)
		{
			m_inputs = inputs;
			m_inputs.push_back(1.); // insert bias value in the vector's tail
		}
		else
		{
			cout << "ERROR: wrong input size for network \n";
			cout << "m_inputs size = " << m_inputs.size() << endl;
			cout << "inputs size   = " << inputs.size() << endl;
			exit(1);
		}


	}

	// Set the same value for all coeffcients
	void Network::set_allWeights(vector<vector<double>>& weights)
	{

		if (weights.size() == m_allweights.size())
		{
			for (int i = 0; i <= m_Nlayers; ++i)
			{
				if (weights[i].size() == m_allweights[i].size())
				{
					m_allweights[i] = weights[i];
					m_network[i].set_allWeights(weights[i]);
				}
				else
				{
					cout << "ERROR : wrong size in network weights, layer id = " << i + 1 << endl;
					cout << "weights[id] size      = " << weights[i].size() << endl;
					cout << "m_allweights[id] size = " << m_allweights[i].size() << endl;
					exit(1);
				}
			}

		}
		else
		{
			cout << "ERROR : wrong size in network weights, number of layers " << endl;
			cout << "weights.size      = " << weights.size() << endl;
			cout << "m_allweights.size = " << m_allweights.size() << endl;
			exit(1);
		}
	}



	// Forward propagation
	void Network::forward_prop()
	{

		vector<double> vect;

		m_network[0].set_inputs(m_inputs);
		m_network[0].compute();

		if (m_prints == 1)
		{
			m_network[0].print_inputs();
			m_network[0].print_outputs();
		}


		for (int i = 1; i <= m_Nlayers; ++i)
		{

			vect = m_network[i - 1].get_outputs();
			//vect.push_back(1.); // adding offset value in vector's tail

			m_network[i].set_inputs(vect);
			m_network[i].compute();

			if (m_prints == 1)
			{
				m_network[i].print_inputs();
				m_network[i].print_outputs();
			}
		}

		m_outputs = m_network[m_Nlayers].get_outputs();
		m_outputs.pop_back(); // removing the bias unit from the outputs of the output layer


	}


	// Get outputs
	vector<double> Network::get_outputs()
	{
		return m_outputs;
	}

	// Get all weights in the network
	vector<vector<double>> Network::get_allWeights()
	{
		return m_allweights;
	}


	// Get outputs of layer id=num
	vector<double> Network::get_layer_outputs(int& num)
	{
		return m_network[num].get_outputs();
	}


	// Get network scheme
	vector<int> Network::get_scheme()
	{
		return m_scheme;
	}


	// get m_cost
	double Network::get_cost()
	{
		return m_cost;
	}


	// Set m_cost
	void Network::set_cost(double cost)
	{
		m_cost = cost;
	}




	// saving the network : scheme and weights
	void Network::save(string path)
	{

		ofstream file(path.c_str());

		if (file)
		{
			for (auto s : m_scheme)
			{
				file << s << " ";
			}

			file << endl;

			for (size_t i = 0; i < m_allweights.size(); ++i)
			{
				for (size_t j = 0; j < m_allweights[i].size(); ++j)
				{
					file << m_allweights[i][j] << " ";
				}
				file << endl;
			}
		}
		else
		{
			cout << "ERROR : in Network.save() , can't open file : " << path.c_str() << endl;
		}
	}


	// Importing a saved network, and building it
	void Network::import(string path, int print)
	{
		ifstream file(path.c_str());

		if (file)
		{

			vector<double> temp;
			string input;
			double w;
			int a;

			//importing network scheme
			getline(file, input);

			stringstream ss(input);

			while (ss >> a)
			{
				m_scheme.push_back(a);
			}

			// building network
			build_network(m_scheme, print);
			m_allweights.clear();

			// reading weights
			while (getline(file, input))
			{
				stringstream ss(input);

				temp.clear();

				while (ss >> w)
				{
					temp.push_back(w);
				}

				m_allweights.push_back(temp);
			}

			// setting network weights
			set_allWeights(m_allweights);

		}
		else
		{
			cout << "ERROR : in Network.import() , can't open file : " << path.c_str() << endl;
		}

	}



	// compute cost function
	void Network::cost_sum(vector<double>& net_outputs, vector<double>& training_outputs)
	{
		//vector<double> net_outputs = m_network.get_outputs();

		for (int i = 0; i < m_scheme.back(); ++i)
		{
			m_cost += -(training_outputs[i] * log(net_outputs[i]) + (1. - training_outputs[i]) * log(1. - net_outputs[i]));
		}
	}


	//compute cost on training set with regularization terms
	void Network::cost(int N_trainings, double lambda)
	{

		double regul_term = 0.;

		for (auto c : m_allweights)
		{
			for (auto w : c)
			{
				regul_term += w*w;
			}
		}

		m_cost = m_cost / ((float)N_trainings) + lambda / ((float)N_trainings * 2.) * regul_term;

	}



}

