neural_network
==============

Tools to build and train neural networks (only supervised learning at the moment)

neuron.h/neuron.cpp
-----------------------
class to create neuron objects, that use a sigmoid function to compute neuron output.


layer.h/layer.cpp
------------------------
class to create layer of neurons.


neural_network.h/neural_network.cpp
-------------------------------------
class to create a neural network as a vector of layers.


back_prop.h/back_prop.cpp
----------------------------
class creating a back_prop object, using a neural_network object.
the back_prop object contains methods to train the neural_network
under supervised learning, using the back propagation algorithm.
