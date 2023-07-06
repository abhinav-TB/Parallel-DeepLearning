#include "neural_network.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <omp.h>

NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes)
    : layers(layerSizes.size()), weights(layerSizes.size() - 1) {
    // Initialize each layer with the specified number of nodes
    for (int i = 0; i < layerSizes.size(); ++i) {
        layers[i] = std::vector<double>(layerSizes[i]);
    }

    // Initialize each weight matrix with random values
    for (int i = 0; i < weights.size(); ++i) {
        weights[i] = std::vector<double>(layerSizes[i] * layerSizes[i+1]);
        for (int j = 0; j < weights[i].size(); ++j) {
            weights[i][j] = (double)rand() / RAND_MAX;  // Random value between 0 and 1
        }
    }
}

void NeuralNetwork::setInputLayer(const std::vector<double>& inputs) {
    layers[0] = inputs;
}

void NeuralNetwork::forwardPropagation() {
    // Parallelize the forward propagation step
    #pragma omp parallel for
    for (int i = 1; i < layers.size(); ++i) {
        for (int j = 0; j < layers[i].size(); ++j) {
            double sum = 0.0;

            // Compute the weighted sum of inputs and apply activation function
            for (int k = 0; k < layers[i-1].size(); ++k) {
                sum += layers[i-1][k] * weights[i-1][j*layers[i-1].size()+k];
            }

            layers[i][j] = sigmoid(sum);
        }
    }
}

void NeuralNetwork::forward_normal(){
    for (int i = 1; i < layers.size(); ++i) {
        for (int j = 0; j < layers[i].size(); ++j) {
            double sum = 0.0;

            // Compute the weighted sum of inputs and apply activation function
            for (int k = 0; k < layers[i-1].size(); ++k) {
                sum += layers[i-1][k] * weights[i-1][j*layers[i-1].size()+k];
            }

            layers[i][j] = sigmoid(sum);
        }
    }
}

double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

void NeuralNetwork::printOutputLayer() {
    std::cout << "Output Layer Values: ";
    for (const auto& value : layers.back()) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}