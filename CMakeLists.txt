cmake_minimum_required(VERSION 3.24)
project(XOR_Artificial_Neural_Network)

set(CMAKE_CXX_STANDARD 17)

add_executable(XOR_Artificial_Neural_Network main.cpp NeuralNet.cpp NeuralNet.h Neuron.cpp)
