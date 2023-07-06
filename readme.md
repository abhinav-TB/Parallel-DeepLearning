# Parallel DeepLearning

This repository contains code for a neural network implementation that uses parallelism to speed up forward propagation. The neural network is implemented in C++ and uses OpenMP for parallelization.

## Requirements

- C++ compiler that supports OpenMP
- OpenMP library

## Usage

1. Clone the repository: `git clone https://github.com/username/Parallel-DeepLearning.git`
2. Navigate to the cloned repository: `cd Parallel-DeepLearning`
3. Compile the code: `g++ -fopenmp main.cpp neural_network.cpp -o main`
4. Run the program: `./main`

## Implementation Details

The neural network is implemented using a vector of layer vectors and a vector of weight matrices. The number of nodes in each layer is specified by the user when creating the neural network object. The weight matrices are initialized with random values between 0 and 1.

The neural network uses forward propagation to make predictions. The input layer is set using the `setInputLayer` method, and the output layer can be obtained using the `getOutputLayer` method. The `feedForward` method performs forward propagation and updates the values in each layer.

Parallelism is achieved using OpenMP. The `feedForward` method is parallelized using the `omp parallel for` directive, which distributes the work across multiple threads.

## Contributing

Contributions are welcome! Please open an issue or pull request for any bug fixes or feature requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
