#include <iostream>
#include <vector>
#include <chrono>
#include "neural_network.hpp"


int main() {
    // Initialize the random seed
    srand(time(NULL));

    // Create a neural network with 10 layers
    NeuralNetwork nn({2,1000,1000,1000,1000,1000,1000,1000,1});

    // Set the input layer values
    nn.setInputLayer({1.0, 0.5});

    // Perform forward propagation without parallelism
    auto start = std::chrono::high_resolution_clock::now();
    nn.forward_normal();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_normal = end - start;
    std::cout << "Forward propagation without parallelism took " << duration_normal.count() << " seconds." << std::endl;
    
    // Perform forward propagation with parallelism
    start = std::chrono::high_resolution_clock::now();
    nn.forwardPropagation();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Forward propagation with parallelism took " << duration.count() << " seconds." << std::endl;
    
    // Calculate and print the speedup
    double speedup = std::abs(duration_normal.count() / duration.count());
    std::cout << "Speedup: " << speedup << std::endl;

    return 0;
}