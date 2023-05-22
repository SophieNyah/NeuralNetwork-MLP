//
// Created by sophie on 10/05/23.
//

#include <cmath>
#include "NeuralNet.h"

double NeuralNet::m_recentAverageSmoothingFactor = 100.0;

NeuralNet::NeuralNet(const std::vector<unsigned int> &topology) {
    std::size_t numLayers = topology.size();

    for (std::size_t i=0; i < numLayers; i++) {
        unsigned int numConnections = (i+1 == numLayers) ? 0 : topology[i+1];
        Layer layer;
        layer.reserve(topology[i]);
        for (int j = 0; j < topology[i]; ++j) {
            layer.push_back(Neuron(numConnections, j));
        }

        m_layers.push_back(layer);
    }
}

void NeuralNet::feedForward(const std::vector<double> &inputVals) {
    for (unsigned int i: inputVals) {
        m_layers[0][i].setOutputValue(inputVals[i]);
    }

    for(unsigned int layerNum=1; layerNum < m_layers.size(); layerNum++) {
        Layer& previousLayer = m_layers[layerNum - 1];
        for (unsigned int n=0; n < m_layers[layerNum].size(); n++) {
            m_layers[layerNum][n].feedForward(previousLayer);
        }
    }
}

void NeuralNet::backPropagation(const std::vector<double> &targetVals) {
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned int n=0; n < outputLayer.size(); n++) {
        double delta = targetVals[n] - outputLayer[n].getOuputValue();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size();
    m_error = sqrt(m_error);

    m_recentAverageError =
            (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
            / (m_recentAverageSmoothingFactor + 1.0);

    for (unsigned n = 0; n < outputLayer.size(); ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    for (unsigned int layerNum = m_layers.size() - 1; layerNum > 0; layerNum--) {
        Layer& layer = m_layers[layerNum];
        Layer& prevLayer = m_layers[layerNum - 1];

        for (unsigned int n=0; n < layer.size(); n++) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void NeuralNet::getResults(std::vector<double> &resultVals) const {
    resultVals.clear();

    for (unsigned int n=0; n < m_layers.back().size(); n++) {
        resultVals.push_back(m_layers.back()[n].getOuputValue());
    }
}
