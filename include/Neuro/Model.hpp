#pragma once

#include "Functions.hpp"
#include "Neuron.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace Neuro {
  static const std::string SAVE_LAYERS_BOUNDARY_DELIMITER = "\x0A\x0A\x0A";
  static const std::string SAVE_LAYERS_DELIMITER = "\x0A\x0A";
  static const std::string SAVE_NEURONS_DELIMITER = "\x0A";
  static const std::string SAVE_NEURONS_ATTRIBUTES_DELIMITER = ";";
  static const std::string SAVE_NEURONS_WEIGHTS_DELIMITER = "|";

  class Model {
  private:
    ERROR_FUNCTION m_error;
    std::vector<std::vector<Neuro::Neuron>> m_neurons;
    uint32_t m_inputsCount;

    bool _IsDelimitersValid() const;

    bool _IsPresentAtIndex(const std::vector<char> &_charArray, const size_t _index, const std::string &_string) const;

  public:
    Model();
    Model(const std::vector<uint32_t> &_layers, const ACTIVATION_FUNCTION _defaultActivationFunction, const ERROR_FUNCTION _errorFunction, const double_t _randomBottom = -1, const double_t _randomTop = 1);

    void Setup(const std::vector<uint32_t> &_layers, const ACTIVATION_FUNCTION _defaultActivationFunction, const ERROR_FUNCTION _errorFunction, const double_t _randomBottom = -1, const double_t _randomTop = 1);

    bool IsValid() const;

    void SetErrorFunction(const ERROR_FUNCTION _errorFunction);
    void SetErrorFunction(const std::string &_errorFunctionString);
    const ERROR_FUNCTION GetErrorFunction() const;
    uint32_t GetInputAmount() const;
    size_t GetLayerCount() const;
    size_t GetLayerSize(const size_t _layer) const;
    Neuro::Neuron &GetNeuron(const size_t _layer, const size_t _index);
    const Neuro::Neuron &GetNeuron(const size_t _layer, const size_t _index) const;

    void Print();
    bool Save(const std::string &_filepath, const std::map<std::string, ACTIVATION_FUNCTION> &_activationDeclarations = Neuro::ACTIVATION_FUNCTIONS_LIST, const std::map<std::string, ERROR_FUNCTION> &_errorDeclarations = Neuro::ERROR_FUNCTION_LIST);
    bool Load(const std::string &_filepath, const std::map<std::string, ACTIVATION_FUNCTION> &_activationDeclarations = Neuro::ACTIVATION_FUNCTIONS_LIST, const std::map<std::string, ERROR_FUNCTION> &_errorDeclarations = Neuro::ERROR_FUNCTION_LIST);
  };
}
