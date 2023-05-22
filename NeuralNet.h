//
// Created by sophie on 10/05/23.
//

#ifndef XOR_ARTIFICIAL_NEURAL_NETWORK_NEURALNET_H
#define XOR_ARTIFICIAL_NEURAL_NETWORK_NEURALNET_H

#include <vector>

class Neuron;
using Layer = std::vector<Neuron>;
struct Connection {
    double weight;
    double deltaWeight;
};

class NeuralNet {

    public:
        explicit NeuralNet(const std::vector<unsigned int>& topology);
        void feedForward(const std::vector<double>& inputVals);
        void backPropagation(const std::vector<double>& targetVals);
        void getResults(std::vector<double>& resultVals) const;

    private:
        std::vector<Layer> m_layers;
        double m_error;
        double m_recentAverageError;
        static double m_recentAverageSmoothingFactor;

};

class Neuron {
    public:
        explicit Neuron(unsigned int numOutputs, unsigned int index_in_layer);
        void setOutputValue(double val);
        double getOuputValue() const;
        void feedForward(const Layer& previousLayer);
        void calcOutputGradients(double targetVal);
        void calcHiddenGradients(const Layer& nextLayer);
        void updateInputWeights(Layer& prevLayer);

    private:
        static double randomWeight();
        static double transferFunction(double x);
        static double transferFunctionDerivative(double x);
        double sumDOW(const Layer& nextLayer) const;

        static double eta;
        static double alpha;

        unsigned int index_in_layer;
        double m_gradient;
        double m_outputVal;
        std::vector<Connection> m_outputWeights;
};


#endif //XOR_ARTIFICIAL_NEURAL_NETWORK_NEURALNET_H
