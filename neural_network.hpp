#pragma once

#include <vector>

class NeuralNetwork {
private:
    std::vector<std::vector<double>> layers;  // Vector of layer vectors
    std::vector<std::vector<double>> weights;  // Vector of weight matrices

public:
    NeuralNetwork(const std::vector<int>& layerSizes);

    void setInputLayer(const std::vector<double>& inputs);

    void forwardPropagation();

    void forward_normal();

    double sigmoid(double x);

    void printOutputLayer();
};