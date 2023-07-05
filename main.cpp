#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>

class NeuralNetwork {
private:
    std::vector<std::vector<double>> layers;  // Vector of layer vectors
    std::vector<std::vector<double>> weights;  // Vector of weight matrices

public:
    NeuralNetwork(const std::vector<int>& layerSizes)
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

    void setInputLayer(const std::vector<double>& inputs) {
        layers[0] = inputs;
    }

    void forwardPropagation() {
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
    void forward_normal(){
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

    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    void printOutputLayer() {
        std::cout << "Output Layer Values: ";
        for (const auto& value : layers.back()) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
};

int main() {
    // Initialize the random seed
    srand(time(NULL));

    // Create a neural network with 10 layers
    NeuralNetwork nn({2,300, 1000, 300, 100, 3000, 1000, 3000, 1000, 100, 100, 20, 3});

    // Set the input layer values
    nn.setInputLayer({1.0, 0.0});

    // Perform forward propagation without parallelism
    auto start = std::chrono::high_resolution_clock::now();
    nn.forward_normal();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Forward propagation without parallelism took " << duration.count() << " seconds." << std::endl;
    
    // Perform forward propagation with parallelism
    start = std::chrono::high_resolution_clock::now();
    nn.forwardPropagation();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Forward propagation with parallelism took " << duration.count() << " seconds." << std::endl;

    return 0;
}