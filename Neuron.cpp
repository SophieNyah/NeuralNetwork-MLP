//
// Created by sophie on 10/05/23.
//

#include <cstdlib>
#include <cmath>
#include "NeuralNet.h"

double Neuron::eta{ 0.15 };
double Neuron::alpha{ 0.5 };

double Neuron::randomWeight() { return rand() / double(RAND_MAX); }

Neuron::Neuron(unsigned int numOutputs, unsigned int index_in_layer) {
    for(unsigned int i=0; i < numOutputs; i++) {
        Connection c{};
        c.weight = Neuron::randomWeight();
        m_outputWeights.push_back(c);
    }

    this->index_in_layer = index_in_layer;
}

void Neuron::setOutputValue(double val) { m_outputVal = val; }

double Neuron::getOuputValue() const { return m_outputVal; }

void Neuron::feedForward(const Layer &previousLayer) {
    double sum = 0.0;

    for(Neuron n: previousLayer) {
        sum += n.getOuputValue() * n.m_outputWeights[index_in_layer].weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}

double Neuron::transferFunction(double x) {
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x) {
    return 1.0 - (x*x);
}

void Neuron::calcOutputGradients(double targetVal) {
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::sumDOW(const Layer &nextLayer) const {
    double sum = 0.0;

    for (unsigned int n=0; n < nextLayer.size(); n++) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer) {
    for (unsigned int n=0; n < prevLayer.size(); n++) {
        Neuron& neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[index_in_layer].deltaWeight;

        double newDeltaWeight = (eta * neuron.getOuputValue() * m_gradient) + (alpha * oldDeltaWeight);
        neuron.m_outputWeights[index_in_layer].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[index_in_layer].weight += newDeltaWeight;
    }
}
