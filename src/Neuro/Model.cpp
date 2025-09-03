#include "Neuro/Model.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Neuro {
bool Model::_IsDelimitersValid() const {
  // Check delimiters
  if (SAVE_LAYERS_DELIMITER == SAVE_NEURONS_DELIMITER) {
    return false;
  }

  if (SAVE_LAYERS_DELIMITER == SAVE_LAYERS_BOUNDARY_DELIMITER) {
    return false;
  }

  return true;
}

bool Model::_IsPresentAtIndex(const std::vector<char> &_charArray,
                              const size_t _index,
                              const std::string &_string) const {
  bool delimiterFound = false;
  for (size_t charIndex = 0; charIndex < _string.length(); charIndex++) {
    if (_charArray[_index + charIndex] != _string[charIndex]) {
      break;
    }

    if (charIndex >= _string.length() - 1) {
      delimiterFound = true;
    }
  }

  return delimiterFound;
}

Model::Model() : m_error(nullptr), m_neurons(), m_inputsCount(0) {}

Model::Model(const std::vector<uint32_t> &_layers,
             const ACTIVATION_FUNCTION _defaultActivationFunction,
             const ERROR_FUNCTION _errorFunction, const double_t _randomBottom,
             const double_t _randomTop) {
  Setup(_layers, _defaultActivationFunction, _errorFunction);
}

void Model::Setup(const std::vector<uint32_t> &_layers,
                  const ACTIVATION_FUNCTION _defaultActivationFunction,
                  const ERROR_FUNCTION _errorFunction,
                  const double_t _randomBottom, const double_t _randomTop) {
  SetErrorFunction(_errorFunction);

  if (_layers.size() > 0) {
    m_inputsCount = _layers[0];
  } else {
    m_inputsCount = 0;
  }

  m_neurons.resize(_layers.size() - 1);

  for (size_t i = 0; i < m_neurons.size(); i++) {
    auto &layer = m_neurons[i];
    layer.resize(_layers[i + 1]);
  }

  for (size_t i = 0; i < _layers.size() - 1; i++) {
    for (size_t j = 0; j < m_neurons[i].size(); j++) {
      m_neurons[i][j].SetInputAmount(_layers[i]);
      m_neurons[i][j].SetActivationFunction(_defaultActivationFunction);
      m_neurons[i][j].Randomize(_randomBottom, _randomTop);
    }
  }
}

bool Model::IsValid() const {
  if (m_error == nullptr) {
    return false;
  }

  if (GetLayerCount() <= 0) {
    return false;
  }

  if (GetInputAmount() <= 0) {
    return false;
  }

  for (size_t layer = 0; layer < GetLayerCount(); layer++) {
    if (layer == 0) {
      for (size_t neuron = 0; neuron < GetLayerSize(layer); neuron++) {
        auto n = GetNeuron(layer, neuron);
        if (n.GetWeightsReference().size() != GetInputAmount()) {
          return false;
        }
      }
    } else {
      for (size_t neuron = 0; neuron < GetLayerSize(layer); neuron++) {
        auto n = GetNeuron(layer, neuron);
        if (n.GetWeightsReference().size() != GetLayerSize(layer - 1)) {
          return false;
        }
      }
    }
  }

  return true;
}

void Model::SetErrorFunction(const ERROR_FUNCTION _errorFunction) {
  if (_errorFunction) {
    m_error = _errorFunction;
  } else {
    throw std::runtime_error("Error function is null");
  }
}

void Model::SetErrorFunction(const std::string &_errorFunctionString) {
  auto found = Neuro::ERROR_FUNCTION_LIST.find(_errorFunctionString)->second;
  if (found) {
    m_error = found;
  } else {
    throw std::runtime_error("Error function string not found");
  }
}

const ERROR_FUNCTION Model::GetErrorFunction() const { return m_error; }

uint32_t Model::GetInputAmount() const { return m_inputsCount; }

size_t Model::GetLayerCount() const { return m_neurons.size(); }

size_t Model::GetLayerSize(const size_t _layer) const {
  return m_neurons[_layer].size();
}

Neuro::Neuron &Model::GetNeuron(const size_t _layer, const size_t _index) {
  return m_neurons[_layer][_index];
}

const Neuro::Neuron &Model::GetNeuron(const size_t _layer,
                                      const size_t _index) const {
  return m_neurons[_layer][_index];
}

void Model::Print() {
  std::cout << "Error: " << (void *)m_error << std::endl;
  std::cout << "Inputs: " << GetInputAmount() << std::endl;
  std::cout << "Layers: ";

  for (size_t i = 0; i < GetLayerCount(); i++) {
    if (i == GetLayerCount() - 1) {
      std::cout << GetLayerSize(i);
    } else {
      std::cout << GetLayerSize(i) << ", ";
    }
  }

  std::cout << std::endl;
}

bool Model::Save(
    const std::string &_filepath,
    const std::map<std::string, ACTIVATION_FUNCTION> &_activationDeclarations,
    const std::map<std::string, ERROR_FUNCTION> &_errorDeclarations) {
  if (!_IsDelimitersValid()) {
    return false;
  }

  if (!IsValid()) {
    return false;
  }

  std::fstream f;
  f.open(_filepath, std::ios::out);

  if (!f.is_open()) {
    f.close();
    return false;
  }

  std::string errorFunctionString;

  for (const auto &errorKeyValuePair : _errorDeclarations) {
    if (errorKeyValuePair.second == m_error) {
      errorFunctionString = errorKeyValuePair.first;
    }
  }

  if (errorFunctionString == "") {
    f.close();
    return false;
  }

  f.write(errorFunctionString.c_str(), errorFunctionString.length());
  f.write(SAVE_LAYERS_BOUNDARY_DELIMITER.c_str(),
          SAVE_LAYERS_BOUNDARY_DELIMITER.length());

  for (size_t layer = 0; layer < GetLayerCount(); layer++) {

    for (size_t neuron = 0; neuron < GetLayerSize(layer); neuron++) {
      auto &n = GetNeuron(layer, neuron);
      std::string activationFunctionString;

      for (const auto &activationKeyValuePair : _activationDeclarations) {
        if (activationKeyValuePair.second == n.GetActivationFunction()) {
          activationFunctionString = activationKeyValuePair.first;
        }
      }

      if (activationFunctionString == "") {
        f.close();
        return false;
      }

      f.write(activationFunctionString.c_str(),
              activationFunctionString.length());
      f.write(SAVE_NEURONS_ATTRIBUTES_DELIMITER.c_str(),
              SAVE_NEURONS_ATTRIBUTES_DELIMITER.length());

      std::string biasString;
      std::string weightsString;

      biasString = std::to_string(n.GetBiasReference());
      f.write(biasString.c_str(), biasString.length());
      f.write(SAVE_NEURONS_ATTRIBUTES_DELIMITER.c_str(),
              SAVE_NEURONS_ATTRIBUTES_DELIMITER.length());

      for (size_t weight = 0; weight < n.GetWeightsReference().size();
           weight++) {
        weightsString += std::to_string(n.GetWeightsReference()[weight]);
        weightsString += SAVE_NEURONS_WEIGHTS_DELIMITER;
      }

      if (weightsString.substr(weightsString.length() -
                               SAVE_NEURONS_WEIGHTS_DELIMITER.length()) ==
          SAVE_NEURONS_WEIGHTS_DELIMITER) {
        weightsString.erase(weightsString.length() -
                            SAVE_NEURONS_WEIGHTS_DELIMITER.length());
      }

      f.write(weightsString.c_str(), weightsString.length());

      if (neuron != GetLayerSize(layer) - 1) {
        f.write(SAVE_NEURONS_DELIMITER.c_str(),
                SAVE_NEURONS_DELIMITER.length());
      }
    }

    if (layer != GetLayerCount() - 1) {
      f.write(SAVE_LAYERS_DELIMITER.c_str(), SAVE_LAYERS_DELIMITER.length());
    }
  }

  f.close();

  return true;
}

bool Model::Load(
    const std::string &_filepath,
    const std::map<std::string, ACTIVATION_FUNCTION> &_activationDeclarations,
    const std::map<std::string, ERROR_FUNCTION> &_errorDeclarations) {
  if (!_IsDelimitersValid()) {
    return false;
  }

  std::vector<std::vector<std::string>> layerNeuronStrings;
  std::vector<char> readBuffer;

  std::fstream f;
  size_t fSize;

  f.open(_filepath, std::ios::in | std::ios::ate);

  if (!f.is_open()) {
    f.close();
    return false;
  }

  fSize = f.tellg();
  f.seekg(f.beg);

  readBuffer.resize(fSize);
  f.read(readBuffer.data(), readBuffer.size());

  f.close();

  size_t index;
  bool delimiterFound;
  std::string errorFunctionString;

  index = 0;
  while (index < readBuffer.size() - SAVE_LAYERS_BOUNDARY_DELIMITER.length()) {
    delimiterFound = _IsPresentAtIndex(readBuffer, index, SAVE_LAYERS_BOUNDARY_DELIMITER);

    if (delimiterFound) {
      break;
    }

    errorFunctionString += readBuffer[index];

    index++;
  }

  SetErrorFunction(errorFunctionString);

  std::string layerString;

  index += SAVE_LAYERS_BOUNDARY_DELIMITER.length();
  while (index < readBuffer.size()) {
    char c = readBuffer[index];

    delimiterFound = _IsPresentAtIndex(readBuffer, index, SAVE_LAYERS_DELIMITER);

    if (delimiterFound || index == readBuffer.size() - 1) {
      std::string neuronString;
      int64_t neuronDelimiterIndex;
      std::vector<std::string> neuronsInLayer;

      neuronDelimiterIndex = layerString.find(SAVE_NEURONS_DELIMITER);

      while (neuronDelimiterIndex != layerString.npos) {
        neuronString = layerString.substr(0, neuronDelimiterIndex);
        layerString.erase(0, neuronDelimiterIndex + SAVE_NEURONS_DELIMITER.length());

        neuronsInLayer.push_back(neuronString);

        neuronDelimiterIndex = layerString.find(SAVE_NEURONS_DELIMITER);
      }

      neuronString = layerString;
      neuronsInLayer.push_back(neuronString);

      layerNeuronStrings.push_back(neuronsInLayer);

      layerString = "";
      index += SAVE_LAYERS_DELIMITER.length();
      continue;
    }

    layerString += readBuffer[index];
    index++;
  }


  m_neurons.resize(layerNeuronStrings.size());
  for (size_t layer = 0; layer < layerNeuronStrings.size(); layer++) {
    m_neurons[layer].resize(layerNeuronStrings[layer].size());

    for (size_t neuron = 0; neuron < layerNeuronStrings[layer].size(); neuron++) {
      size_t delimiterIndex;

      std::string neuronString = layerNeuronStrings[layer][neuron];
      delimiterIndex = neuronString.find(SAVE_NEURONS_ATTRIBUTES_DELIMITER);
      std::string activationFunctionString = neuronString.substr(0, delimiterIndex);
      neuronString.erase(0, delimiterIndex + SAVE_NEURONS_ATTRIBUTES_DELIMITER.length());

      GetNeuron(layer, neuron).SetActivationFunction(activationFunctionString);

      delimiterIndex = neuronString.find(SAVE_NEURONS_ATTRIBUTES_DELIMITER);
      std::string biasString = neuronString.substr(0, delimiterIndex);
      neuronString.erase(0, delimiterIndex + SAVE_NEURONS_ATTRIBUTES_DELIMITER.length());

      GetNeuron(layer, neuron).GetBiasReference() = std::stod(biasString);

      delimiterIndex = neuronString.find(SAVE_NEURONS_WEIGHTS_DELIMITER);
      while (delimiterIndex != neuronString.npos) {
        std::string weightString = neuronString.substr(0, delimiterIndex);
        neuronString.erase(0, delimiterIndex + SAVE_NEURONS_WEIGHTS_DELIMITER.length());

        GetNeuron(layer, neuron).GetWeightsReference().push_back(std::stod(weightString));

        delimiterIndex = neuronString.find(SAVE_NEURONS_WEIGHTS_DELIMITER);
      }

      std::string weightString = neuronString;
      GetNeuron(layer, neuron).GetWeightsReference().push_back(std::stod(weightString));
    }
  }

  if (m_neurons.size() > 0 && m_neurons[0].size() > 0) {
    m_inputsCount = GetNeuron(0, 0).GetInputAmount();
  }
  else {
    m_inputsCount = 0;
  }

  return true;
}

} // namespace Neuro
