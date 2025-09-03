#include "Neuro/Model.hpp"
#include "Neuro/Network.hpp"
#include <cmath>
#include <iostream>
#include <vector>

int main(int argc, char *argv[]) {
  Neuro::Model m;

  const std::vector<std::vector<double_t>> training_inputs = {
      {0, 1}, {1, 0}, {0, 0}, {1, 1}};

  const std::vector<std::vector<double_t>> training_outputs = {
      {1}, {1}, {0}, {0}};

  if (!m.Load("test.mdl") || !m.IsValid()) {
    m.Setup({2, 2, 1}, Neuro::A_SIGMOID, Neuro::C_SQUARED_ERROR);
  }

  Neuro::Network::Train(500000, m, training_inputs, training_outputs, 0.01);
  std::cout << std::endl;

  std::vector<double_t> output;

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

  m.Save("test.mdl");

  return 0;
}
