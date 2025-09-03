# Neuro Library

## Table of Content
1. [Information about the library](#information)  
2. [Building and Linking](#building-and-linking) 
3. [Usage](#usage)  
4. [Licensing](#licensing)

## Information
This is a simple C++ library to use Neural Networks in C++. 
This library is written for the purpose to teach myself the basics
of Neural Networks and Backward Propagation. It is also quite fast
and supports any configuration of layers and neurons.

## Building and Linking
This library uses CMake to handle building. If using as a library 
in another project, use `add_subdirectory` in CMake to include it 
in the build cycle. To link use `target_link_libraries` with your 
target and link `Neuro`.

## Usage
See `src/Neuro-Test/main.cpp` for a simple example.

```c++
#include "Neuro/Model.hpp"
#include "Neuro/Network.hpp"
#include <cmath>
#include <iostream>
#include <vector>

int main(int argc, char *argv[]) {
  // Create a model object
  Neuro::Model m;

  // Training data
  const std::vector<std::vector<double_t>> training_inputs = {
      {0, 1}, {1, 0}, {0, 0}, {1, 1}};

  // Training data output
  const std::vector<std::vector<double_t>> training_outputs = {
      {1}, {1}, {0}, {0}};

  // Try to load 'test.mdl' model file else setup a new model
  if (!m.Load("test.mdl") || !m.IsValid()) {
    m.Setup({2, 2, 1}, Neuro::A_SIGMOID, Neuro::C_SQUARED_ERROR);
  }

  // Train the model agains the training data for 500000 epochs and a learning rate of 0.01
  Neuro::Network::Train(500000, m, training_inputs, training_outputs, 0.01);
  std::cout << std::endl;

  std::vector<double_t> output;

  // Test output after training and output all values
  // Note: This example implements a XOR algorithm
  output = Neuro::Network::Calculate(m, {0, 1});
  for (size_t out = 0; out < output.size(); out++) {
    std::cout << output[0] << ' ';
  }
  std::cout << std::endl;

  output = Neuro::Network::Calculate(m, {1, 0});
  for (size_t out = 0; out < output.size(); out++) {
    std::cout << output[0] << ' ';
  }
  std::cout << std::endl;

  output = Neuro::Network::Calculate(m, {0, 0});
  for (size_t out = 0; out < output.size(); out++) {
    std::cout << output[0] << ' ';
  }
  std::cout << std::endl;

  output = Neuro::Network::Calculate(m, {1, 1});
  for (size_t out = 0; out < output.size(); out++) {
    std::cout << output[0] << ' ';
  }
  std::cout << std::endl;

  // Save model after training
  m.Save("test.mdl");

  return 0;
}
```

## Licensing
You're in luck. This library does not use any external libraries. The only applicable 
license is stored within the top level license file.
