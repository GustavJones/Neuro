#pragma once
#include "Functions.hpp"
#include <cmath>
#include <cstddef>
#include <vector>
#include <mutex>

namespace Neuro {
  class Neuron {
  private:
    mutable std::mutex m_accessing;
    ACTIVATION_FUNCTION m_activation;
    std::vector<double_t> m_weights;
    double_t m_bias;

  public:
    Neuron();
    Neuron(Neuron&& _obj);
    Neuron(const Neuron& _obj);
    Neuron& operator=(Neuron&& _obj);
    Neuron& operator=(const Neuron& _obj);

    void SetActivationFunction(const std::string &_funcString);
    void SetActivationFunction(const ACTIVATION_FUNCTION _func);
    const ACTIVATION_FUNCTION GetActivationFunction() const;

    size_t GetInputAmount() const;
    void SetInputAmount(const size_t _amount);

    double_t GetWeight(const size_t _index) const;
    void SetWeight(const size_t _index, const double_t _value);
    const size_t GetWeightsAmount() const;
    void SetWeightsAmount(const size_t _newsize);

    void PushBackWeight(const double_t _value);

    double_t GetBias() const;
    void SetBias(const double_t _value);

    void Randomize(const double_t _bottomRange, const double_t _topRange);
    double_t Calculate(const std::vector<double_t> &_input, bool _activated = true) const;
    void Print() const;
  };
}
