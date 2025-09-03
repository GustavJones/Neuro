#pragma once
#include "Functions.hpp"
#include <cmath>
#include <cstddef>
#include <vector>

namespace Neuro {
  class Neuron {
  private:
    ACTIVATION_FUNCTION m_activation;
    std::vector<double_t> m_weights;
    double_t m_bias;

  public:
    void SetActivationFunction(const std::string &_funcString);
    void SetActivationFunction(const ACTIVATION_FUNCTION _func);
    const ACTIVATION_FUNCTION GetActivationFunction() const;

    size_t GetInputAmount() const;
    void SetInputAmount(const size_t _amount);

    std::vector<double_t> &GetWeightsReference();
    const std::vector<double_t> &GetWeightsReference() const;
    double_t &GetBiasReference();
    const double_t &GetBiasReference() const;

    void Randomize(const double_t _bottomRange, const double_t _topRange);
    double_t Calculate(const std::vector<double_t> &_input, bool _activated = true) const;
    void Print() const;
  };
}
