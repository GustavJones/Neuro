#include "Neuro/Neuron.hpp"
#include <ctime>
#include <iostream>
#include <random>
#include <stdexcept>

namespace Neuro {
void Neuron::SetActivationFunction(const std::string &_funcString) {
  auto found = Neuro::ACTIVATION_FUNCTIONS_LIST.find(_funcString)->second;
  if (found) {
    m_activation = found;
  }
  else {
    throw std::runtime_error("Activation function string not found");
  }
}

void Neuron::SetActivationFunction(const ACTIVATION_FUNCTION _func) {
  if (_func) {
    m_activation = _func;
  }
  else {
    throw std::runtime_error("Activation function is null");
  }
}

const ACTIVATION_FUNCTION Neuron::GetActivationFunction() const {
  return m_activation;
}

size_t Neuron::GetInputAmount() const { return m_weights.size(); }

void Neuron::SetInputAmount(const size_t _amount) { m_weights.resize(_amount); }

std::vector<double_t> &Neuron::GetWeightsReference() { return m_weights; }

const std::vector<double_t> &Neuron::GetWeightsReference() const {
  return m_weights;
}

double_t &Neuron::GetBiasReference() { return m_bias; }

const double_t &Neuron::GetBiasReference() const { return m_bias; }

double_t Neuron::Calculate(const std::vector<double_t> &_input,
                           bool _activated) const {
  double_t output;

  if (_input.size() != m_weights.size()) {
    throw std::runtime_error("Inputs does not match weights");
  }

  output = 0;
  for (size_t i = 0; i < m_weights.size(); i++) {
    output += m_weights[i] * _input[i];
  }

  output += m_bias;

  if (_activated) {
    return m_activation(output, false);
  } else {
    return output;
  }
}

void Neuron::Randomize(const double_t _bottomRange, const double_t _topRange) {
  std::default_random_engine generator(time(nullptr));
  std::uniform_real_distribution<double_t> distribution(_bottomRange,
                                                        _topRange);

  for (size_t weight = 0; weight < m_weights.size(); weight++) {
    m_weights[weight] = distribution(generator);
  }

  m_bias = distribution(generator);
}

void Neuron::Print() const {
  // std::cout << "Neuron" << std::endl;
  std::cout << "Activation: " << (void *)m_activation << std::endl;
  std::cout << "Bias: " << GetBiasReference() << std::endl;

  std::cout << "Weights: ";

  for (size_t i = 0; i < GetInputAmount(); i++) {
    if (i == GetInputAmount() - 1) {
      std::cout << GetWeightsReference()[i];
    } else {
      std::cout << GetWeightsReference()[i] << ", ";
    }
  }

  std::cout << std::endl;
}

} // namespace Neuro
