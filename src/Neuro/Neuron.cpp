#include "Neuro/Neuron.hpp"
#include <ctime>
#include <iostream>
#include <random>
#include <stdexcept>

namespace Neuro {
Neuron::Neuron() : m_bias(0), m_activation(nullptr), m_weights(0), m_accessing() {}

Neuron::Neuron(Neuron&& _obj) {
  SetActivationFunction(_obj.GetActivationFunction());
  SetBias(_obj.GetBias());

  SetWeightsAmount(_obj.GetWeightsAmount());

  for (size_t weight = 0; weight < _obj.GetWeightsAmount(); weight++)
  {
    SetWeight(weight, _obj.GetWeight(weight));
  }
}

Neuron::Neuron(const Neuron& _obj) {
  SetActivationFunction(_obj.GetActivationFunction());
  SetBias(_obj.GetBias());

  SetWeightsAmount(_obj.GetWeightsAmount());

  for (size_t weight = 0; weight < _obj.GetWeightsAmount(); weight++)
  {
    SetWeight(weight, _obj.GetWeight(weight));
  }
}

Neuron& Neuron::operator=(Neuron&& _obj) {
  SetActivationFunction(_obj.GetActivationFunction());
  SetBias(_obj.GetBias());

  SetWeightsAmount(_obj.GetWeightsAmount());

  for (size_t weight = 0; weight < _obj.GetWeightsAmount(); weight++)
  {
    SetWeight(weight, _obj.GetWeight(weight));
  }

  return *this;
}

Neuron& Neuron::operator=(const Neuron& _obj) {
  SetActivationFunction(_obj.GetActivationFunction());
  SetBias(_obj.GetBias());

  SetWeightsAmount(_obj.GetWeightsAmount());

  for (size_t weight = 0; weight < _obj.GetWeightsAmount(); weight++)
  {
    SetWeight(weight, _obj.GetWeight(weight));
  }

  return *this;
}

void Neuron::SetActivationFunction(const std::string &_funcString) {
  m_accessing.lock();
  auto found = Neuro::ACTIVATION_FUNCTIONS_LIST.find(_funcString)->second;
  if (found) {
    m_activation = found;
  }
  else {
    m_accessing.unlock();
    throw std::runtime_error("Activation function string not found");
  }

  m_accessing.unlock();
}

void Neuron::SetActivationFunction(const ACTIVATION_FUNCTION _func) {
  m_accessing.lock();
  if (_func) {
    m_activation = _func;
  }
  else {
    m_accessing.unlock();
    throw std::runtime_error("Activation function is null");
  }

  m_accessing.unlock();
}

const ACTIVATION_FUNCTION Neuron::GetActivationFunction() const {
  m_accessing.lock();
  auto out = m_activation;
  m_accessing.unlock();

  return out;
}

size_t Neuron::GetInputAmount() const { 
  m_accessing.lock();
  auto out = m_weights.size();
  m_accessing.unlock();

  return out; 
}

void Neuron::SetInputAmount(const size_t _amount) { 
  m_accessing.lock();
  m_weights.resize(_amount); 
  m_accessing.unlock();
}

double_t Neuron::GetWeight(const size_t _index) const {
  m_accessing.lock();
  auto out = m_weights[_index];
  m_accessing.unlock();

  return out;
}

void Neuron::SetWeight(const size_t _index, const double_t _value) {
  m_accessing.lock();
  m_weights[_index] = _value;
  m_accessing.unlock();
}

const size_t Neuron::GetWeightsAmount() const {
  m_accessing.lock();
  auto out = m_weights.size();
  m_accessing.unlock();

  return out;
}

void Neuron::SetWeightsAmount(const size_t _newsize) {
  m_accessing.lock();
  m_weights.resize(_newsize);
  m_accessing.unlock();
}

void Neuron::PushBackWeight(const double_t _value) {
  m_accessing.lock();
  m_weights.push_back(_value);
  m_accessing.unlock();
}

double_t Neuron::GetBias() const {
  m_accessing.lock();
  auto out = m_bias;
  m_accessing.unlock();

  return out;
}

void Neuron::SetBias(const double_t _value) {
  m_accessing.lock();
  m_bias = _value;
  m_accessing.unlock();
}

double_t Neuron::Calculate(const std::vector<double_t> &_input, bool _activated) const {
  double_t output;

  if (_input.size() != GetWeightsAmount()) {
    throw std::runtime_error("Inputs does not match weights");
  }

  output = 0;
  for (size_t i = 0; i < GetWeightsAmount(); i++) {
    m_accessing.lock();
    output += m_weights[i] * _input[i];
    m_accessing.unlock();
  }

  output += GetBias();

  if (_activated) {
    return GetActivationFunction()(output, false);
  } else {
    return output;
  }
}

void Neuron::Randomize(const double_t _bottomRange, const double_t _topRange) {
  std::default_random_engine generator(time(nullptr));
  std::uniform_real_distribution<double_t> distribution(_bottomRange,
                                                        _topRange);

  for (size_t weight = 0; weight < GetWeightsAmount(); weight++) {
    m_accessing.lock();
    m_weights[weight] = distribution(generator);
    m_accessing.unlock();
  }

  m_accessing.lock();
  m_bias = distribution(generator);
  m_accessing.unlock();
}

void Neuron::Print() const {
  std::cout << "Activation: " << (void *)GetActivationFunction() << std::endl;
  std::cout << "Bias: " << GetBias() << std::endl;

  std::cout << "Weights: ";

  for (size_t i = 0; i < GetInputAmount(); i++) {
    if (i == GetInputAmount() - 1) {
      std::cout << GetWeight(i);
    } else {
      std::cout << GetWeight(i) << ", ";
    }
  }

  std::cout << std::endl;
}

} // namespace Neuro
