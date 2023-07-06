#include <iostream>
#include <vector>
#include <chrono>
#include "neural_network.hpp"

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