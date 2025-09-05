#include "Neuro/Model.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static std::mutex delimitersMutex;

namespace Neuro {
bool Model::_IsDelimitersValid() {
  // Check delimiters
  delimitersMutex.lock();
  if (SAVE_LAYERS_DELIMITER == SAVE_NEURONS_DELIMITER) {
    delimitersMutex.unlock();
    return false;
  }
  
  if (SAVE_LAYERS_DELIMITER == SAVE_LAYERS_BOUNDARY_DELIMITER) {
    delimitersMutex.unlock();
    return false;
  }
  delimitersMutex.unlock();

  return true;
}

bool Model::_IsPresentAtIndex(const std::vector<char> &_charArray,
                              const size_t _index,
                              const std::string &_string) {
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

Model::Model() : m_error(nullptr), m_neurons(), m_inputsCount(0), m_accessing() {}

Model::Model(const std::vector<uint32_t> &_layers,
             const ACTIVATION_FUNCTION _defaultActivationFunction,
             const ERROR_FUNCTION _errorFunction, const double_t _randomBottom,
             const double_t _randomTop) {
  Setup(_layers, _defaultActivationFunction, _errorFunction);
}

Model::Model(Model&& _obj) {
  SetErrorFunction(_obj.GetErrorFunction());

  m_accessing.lock();
  m_neurons.resize(_obj.GetLayerCount());
  for (size_t layer = 0; layer < _obj.GetLayerCount(); layer++)
  {
    m_neurons[layer].resize(_obj.GetLayerSize(layer));

    for (size_t neuron = 0; neuron < _obj.GetLayerSize(layer); neuron++)
    {
      m_neurons[layer][neuron] = _obj.GetNeuron(layer, neuron);
    }
  }

  m_inputsCount = _obj.GetInputAmount();

  m_accessing.unlock();
}

Model::Model(const Model& _obj) {
  SetErrorFunction(_obj.GetErrorFunction());

  m_accessing.lock();
  m_neurons.resize(_obj.GetLayerCount());
  for (size_t layer = 0; layer < _obj.GetLayerCount(); layer++)
  {
    m_neurons[layer].resize(_obj.GetLayerSize(layer));

    for (size_t neuron = 0; neuron < _obj.GetLayerSize(layer); neuron++)
    {
      m_neurons[layer][neuron] = _obj.GetNeuron(layer, neuron);
    }
  }

  m_inputsCount = _obj.GetInputAmount();

  m_accessing.unlock();
}

Model& Model::operator=(Model&& _obj) {
  SetErrorFunction(_obj.GetErrorFunction());

  m_accessing.lock();
  m_neurons.resize(_obj.GetLayerCount());
  for (size_t layer = 0; layer < _obj.GetLayerCount(); layer++)
  {
    m_neurons[layer].resize(_obj.GetLayerSize(layer));

    for (size_t neuron = 0; neuron < _obj.GetLayerSize(layer); neuron++)
    {
      m_neurons[layer][neuron] = _obj.GetNeuron(layer, neuron);
    }
  }

  m_inputsCount = _obj.GetInputAmount();

  m_accessing.unlock();
  return *this;
}

Model& Model::operator=(const Model& _obj) {
  SetErrorFunction(_obj.GetErrorFunction());

  m_accessing.lock();
  m_neurons.resize(_obj.GetLayerCount());
  for (size_t layer = 0; layer < _obj.GetLayerCount(); layer++)
  {
    m_neurons[layer].resize(_obj.GetLayerSize(layer));

    for (size_t neuron = 0; neuron < _obj.GetLayerSize(layer); neuron++)
    {
      m_neurons[layer][neuron] = _obj.GetNeuron(layer, neuron);
    }
  }

  m_inputsCount = _obj.GetInputAmount();

  m_accessing.unlock();
  return *this;
}

void Model::Setup(const std::vector<uint32_t> &_layers,
                  const ACTIVATION_FUNCTION _defaultActivationFunction,
                  const ERROR_FUNCTION _errorFunction,
                  const double_t _randomBottom, const double_t _randomTop) {
  SetErrorFunction(_errorFunction);

  m_accessing.lock();
  if (_layers.size() > 0) {
    m_inputsCount = _layers[0];
  } else {
    m_inputsCount = 0;
  }

  m_neurons.resize(_layers.size() - 1);
  m_accessing.unlock();

  for (size_t i = 0; i < m_neurons.size(); i++) {
    m_accessing.lock();
    auto &layer = m_neurons[i];
    layer.resize(_layers[i + 1]);
    m_accessing.unlock();
  }

  m_accessing.lock();
  for (size_t i = 0; i < _layers.size() - 1; i++) {
    for (size_t j = 0; j < m_neurons[i].size(); j++) {
      m_neurons[i][j].SetInputAmount(_layers[i]);
      m_neurons[i][j].SetActivationFunction(_defaultActivationFunction);
      m_neurons[i][j].Randomize(_randomBottom, _randomTop);
    }
  }
  m_accessing.unlock();
}

bool Model::IsValid() const {
  m_accessing.lock();

  if (m_error == nullptr) {
    m_accessing.unlock();
    return false;
  }

  m_accessing.unlock();

  if (GetLayerCount() <= 0) {
    return false;
  }

  if (GetInputAmount() <= 0) {
    return false;
  }

  for (size_t layer = 0; layer < GetLayerCount(); layer++) {
    if (layer == 0) {
      for (size_t neuron = 0; neuron < GetLayerSize(layer); neuron++) {
        auto &n = GetNeuron(layer, neuron);
        if (n.GetWeightsAmount() != GetInputAmount()) {
          return false;
        }
      }
    } else {
      for (size_t neuron = 0; neuron < GetLayerSize(layer); neuron++) {
        auto &n = GetNeuron(layer, neuron);
        if (n.GetWeightsAmount() != GetLayerSize(layer - 1)) {
          return false;
        }
      }
    }
  }

  return true;
}

void Model::SetErrorFunction(const ERROR_FUNCTION _errorFunction) {
  m_accessing.lock();
  if (_errorFunction) {
    m_error = _errorFunction;
  } else {
    m_accessing.unlock();
    throw std::runtime_error("Error function is null");
  }

  m_accessing.unlock();
}

void Model::SetErrorFunction(const std::string &_errorFunctionString) {
  m_accessing.lock();
  auto found = Neuro::ERROR_FUNCTION_LIST.find(_errorFunctionString)->second;
  if (found) {
    m_error = found;
  } else {
    m_accessing.unlock();
    throw std::runtime_error("Error function string not found");
  }

  m_accessing.unlock();
}

const ERROR_FUNCTION Model::GetErrorFunction() const { 
  m_accessing.lock();
  auto out = m_error;
  m_accessing.unlock();
  return out; 
}

uint32_t Model::GetInputAmount() const { 
  m_accessing.lock();
  auto out = m_inputsCount;
  m_accessing.unlock();
  return out; 
}

size_t Model::GetLayerCount() const { 
  m_accessing.lock();
  auto out = m_neurons.size();
  m_accessing.unlock();
  return out; 
}

size_t Model::GetLayerSize(const size_t _layer) const {
  m_accessing.lock();
  auto out = m_neurons[_layer].size();
  m_accessing.unlock();
  return out;
}

Neuro::Neuron &Model::GetNeuron(const size_t _layer, const size_t _index) {
  m_accessing.lock();
  auto& out = m_neurons[_layer][_index];
  m_accessing.unlock();
  return out;
}

const Neuro::Neuron &Model::GetNeuron(const size_t _layer,
                                      const size_t _index) const {
  m_accessing.lock();
  auto& out = m_neurons[_layer][_index];
  m_accessing.unlock();
  return out;
}

void Model::Print() {
  const auto errorFunction = (void*)GetErrorFunction();
  const auto inputs = GetInputAmount();
  const auto layerCount = GetLayerCount();

  std::vector<size_t> layerSizes(layerCount);

  for (size_t i = 0; i < layerCount; i++)
  {
    layerSizes[i] = GetLayerSize(i);
  }

  m_accessing.lock();
  std::cout << "Error: " << errorFunction << std::endl;
  std::cout << "Inputs: " << inputs << std::endl;
  std::cout << "Layers: ";

  for (size_t i = 0; i < layerCount; i++) {
    if (i == layerCount - 1) {
      std::cout << layerSizes[i];
    } else {
      std::cout << layerSizes[i] << ", ";
    }
  }

  std::cout << std::endl;

  m_accessing.unlock();
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
    if (errorKeyValuePair.second == GetErrorFunction()) {
      errorFunctionString = errorKeyValuePair.first;
    }
  }

  if (errorFunctionString == "") {
    f.close();
    return false;
  }

  f.write(errorFunctionString.c_str(), errorFunctionString.length());
  delimitersMutex.lock();
  f.write(SAVE_LAYERS_BOUNDARY_DELIMITER.c_str(),
          SAVE_LAYERS_BOUNDARY_DELIMITER.length());
  delimitersMutex.unlock();

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
      delimitersMutex.lock();
      f.write(SAVE_NEURONS_ATTRIBUTES_DELIMITER.c_str(),
              SAVE_NEURONS_ATTRIBUTES_DELIMITER.length());
      delimitersMutex.unlock();

      std::string biasString;
      std::string weightsString;

      biasString = std::to_string(n.GetBias());
      f.write(biasString.c_str(), biasString.length());
      delimitersMutex.lock();
      f.write(SAVE_NEURONS_ATTRIBUTES_DELIMITER.c_str(),
              SAVE_NEURONS_ATTRIBUTES_DELIMITER.length());
      delimitersMutex.unlock();

      for (size_t weight = 0; weight < n.GetWeightsAmount();
           weight++) {
        weightsString += std::to_string(n.GetWeight(weight));
        delimitersMutex.lock();
        weightsString += SAVE_NEURONS_WEIGHTS_DELIMITER;
        delimitersMutex.unlock();
      }

      delimitersMutex.lock();
      if (weightsString.substr(weightsString.length() -
                               SAVE_NEURONS_WEIGHTS_DELIMITER.length()) ==
          SAVE_NEURONS_WEIGHTS_DELIMITER) {
        weightsString.erase(weightsString.length() -
                            SAVE_NEURONS_WEIGHTS_DELIMITER.length());
      }
      delimitersMutex.unlock();

      f.write(weightsString.c_str(), weightsString.length());

      if (neuron != GetLayerSize(layer) - 1) {
        delimitersMutex.lock();
        f.write(SAVE_NEURONS_DELIMITER.c_str(),
                SAVE_NEURONS_DELIMITER.length());
        delimitersMutex.unlock();
      }
    }

    if (layer != GetLayerCount() - 1) {
      delimitersMutex.lock();
      f.write(SAVE_LAYERS_DELIMITER.c_str(), SAVE_LAYERS_DELIMITER.length());
      delimitersMutex.unlock();
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
  delimitersMutex.lock();
  const size_t saveLayersBoundaryDelimiterLength = SAVE_LAYERS_BOUNDARY_DELIMITER.length();
  delimitersMutex.unlock();

  while (index < readBuffer.size() - saveLayersBoundaryDelimiterLength) {
    delimitersMutex.lock();
    delimiterFound = _IsPresentAtIndex(readBuffer, index, SAVE_LAYERS_BOUNDARY_DELIMITER);
    delimitersMutex.unlock();

    if (delimiterFound) {
      break;
    }

    errorFunctionString += readBuffer[index];

    index++;
  }

  SetErrorFunction(errorFunctionString);

  std::string layerString;

  delimitersMutex.lock();
  index += SAVE_LAYERS_BOUNDARY_DELIMITER.length();
  delimitersMutex.unlock();

  while (index < readBuffer.size()) {
    char c = readBuffer[index];

    delimitersMutex.lock();
    delimiterFound = _IsPresentAtIndex(readBuffer, index, SAVE_LAYERS_DELIMITER);
    delimitersMutex.unlock();

    if (delimiterFound || index == readBuffer.size() - 1) {
      std::string neuronString;
      int64_t neuronDelimiterIndex;
      std::vector<std::string> neuronsInLayer;

      delimitersMutex.lock();
      neuronDelimiterIndex = layerString.find(SAVE_NEURONS_DELIMITER);
      delimitersMutex.unlock();

      while (neuronDelimiterIndex != layerString.npos) {
        neuronString = layerString.substr(0, neuronDelimiterIndex);

        delimitersMutex.lock();
        layerString.erase(0, neuronDelimiterIndex + SAVE_NEURONS_DELIMITER.length());
        delimitersMutex.unlock();

        neuronsInLayer.push_back(neuronString);

        delimitersMutex.lock();
        neuronDelimiterIndex = layerString.find(SAVE_NEURONS_DELIMITER);
        delimitersMutex.unlock();
      }

      neuronString = layerString;
      neuronsInLayer.push_back(neuronString);

      layerNeuronStrings.push_back(neuronsInLayer);

      layerString = "";
      delimitersMutex.lock();
      index += SAVE_LAYERS_DELIMITER.length();
      delimitersMutex.unlock();
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

      GetNeuron(layer, neuron).SetBias(std::stod(biasString));

      delimiterIndex = neuronString.find(SAVE_NEURONS_WEIGHTS_DELIMITER);
      while (delimiterIndex != neuronString.npos) {
        std::string weightString = neuronString.substr(0, delimiterIndex);
        neuronString.erase(0, delimiterIndex + SAVE_NEURONS_WEIGHTS_DELIMITER.length());

        GetNeuron(layer, neuron).PushBackWeight(std::stod(weightString));

        delimiterIndex = neuronString.find(SAVE_NEURONS_WEIGHTS_DELIMITER);
      }

      std::string weightString = neuronString;
      GetNeuron(layer, neuron).PushBackWeight(std::stod(weightString));
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
