#include "Neuro/Network.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace Neuro {
void Network::CalculateStructures(const Model &_model, const std::vector<double_t> &_inputs, UnactivatedStructure &_unactivated, ActivatedStructure &_activated) {
  _unactivated.GetValues().resize(_model.GetLayerCount());
  _activated.GetValues().resize(_model.GetLayerCount());

  for (size_t layer = 0; layer < _model.GetLayerCount(); layer++) {
    _unactivated[layer].GetValues().resize(_model.GetLayerSize(layer));
    _activated[layer].GetValues().resize(_model.GetLayerSize(layer));
  }

  std::vector<double_t> previousLayerOutputs = _inputs;
  std::vector<double_t> layerOutputs;

  for (size_t layer = 0; layer < _model.GetLayerCount(); layer++) {
    layerOutputs.resize(_model.GetLayerSize(layer));

    for (size_t neuron = 0; neuron < _model.GetLayerSize(layer); neuron++) {
      _unactivated[layer][neuron] = _model.GetNeuron(layer, neuron)
                                        .Calculate(previousLayerOutputs, false);
      _activated[layer][neuron] =
          _model.GetNeuron(layer, neuron)
              .GetActivationFunction()(_unactivated[layer][neuron], false);
      layerOutputs[neuron] = _activated[layer][neuron];
    }

    previousLayerOutputs = layerOutputs;
  }
}

double_t Network::CalculateMeanError(Model &_model, const std::vector<double_t> &_inputs, const std::vector<double_t> &_expectedOutputs) {
  auto output = Calculate(_model, _inputs);
  double_t total = 0;

  for (size_t i = 0; i < output.size(); i++) {
    total += _model.GetErrorFunction()(output[i], _expectedOutputs[i], false);
  }

  return total / output.size();
}

double_t Network::CalculateBatchMeanError(
    Model &_model, const std::vector<std::vector<double_t>> &_batchInputs,
    const std::vector<std::vector<double_t>> &_batchExpectedOutputs) {
  double_t total = 0;

  for (size_t i = 0; i < _batchInputs.size(); i++) {
    total +=
        CalculateMeanError(_model, _batchInputs[i], _batchExpectedOutputs[i]);
  }

  total /= _batchInputs.size();
  return total;
}

std::vector<double_t> Network::Calculate(const Model &_model,
                                        const std::vector<double_t> &_inputs) {
  if (!_model.IsValid()) {
    throw std::runtime_error("Model not loaded correctly");
  }

  UnactivatedStructure us;
  ActivatedStructure as;

  CalculateStructures(_model, _inputs, us, as);

  return as[_model.GetLayerCount() - 1].GetValues();
}

void Network::BackwardPropagate(Model &_model,
                                const std::vector<double_t> &_inputs,
                                const std::vector<double_t> &_expected,
                                const double_t _learningRate) {
  DeltaStructure ds;

  UnactivatedStructure us;
  ActivatedStructure as;
  CalculateStructures(_model, _inputs, us, as);

  ds.GetValues().resize(_model.GetLayerCount());
  for (size_t layer = 0; layer < _model.GetLayerCount(); layer++) {
    ds.GetValues()[layer].GetValues().resize(_model.GetLayerSize(layer));
  }

  double_t delta, deltaSum;

  for (int64_t layer = _model.GetLayerCount() - 1; layer >= 0; layer--) {
    for (size_t neuron = 0; neuron < _model.GetLayerSize(layer); neuron++) {
      Neuro::Neuron &n = _model.GetNeuron(layer, neuron);

      // Last Layer and not first layer
      if (layer == _model.GetLayerCount() - 1 && layer > 0) {
        delta = 1;
        delta /= _model.GetLayerSize(_model.GetLayerCount() - 1);
        delta *= _model.GetErrorFunction()(as[layer][neuron], _expected[neuron], true);
        delta *= n.GetActivationFunction()(us[layer][neuron], true);

        // Weights
        for (size_t weight = 0; weight < n.GetWeightsReference().size(); weight++) {
          n.GetWeightsReference()[weight] -= (as[layer - 1][weight] * delta) * _learningRate;
        }

        // Bias
        n.GetBiasReference() -= delta * _learningRate;

        ds[layer][neuron] = delta;
      }
      // Last layer and first layer
      else if (layer == _model.GetLayerCount() - 1 && layer == 0) {
        delta = 1;
        delta /= _model.GetLayerSize(_model.GetLayerCount() - 1);
        delta *= _model.GetErrorFunction()(as[layer][neuron], _expected[neuron], true);
        delta *= n.GetActivationFunction()(us[layer][neuron], true);

        // Weights
        for (size_t weight = 0; weight < n.GetWeightsReference().size(); weight++) {
          n.GetWeightsReference()[weight] -= (_inputs[weight] * delta) * _learningRate;
        }

        // Bias
        n.GetBiasReference() -= delta * _learningRate;

        ds[layer][neuron] = delta;
      }
      // Other layer and not first layer
      else if (layer != _model.GetLayerCount() - 1 && layer > 0) {
        delta = 1;
        deltaSum = 0;
        for (size_t nextLayerNeuron = 0; nextLayerNeuron < _model.GetLayerSize(layer + 1); nextLayerNeuron++) {
          deltaSum += ds[layer + 1][nextLayerNeuron] * _model.GetNeuron(layer + 1, nextLayerNeuron).GetWeightsReference()[neuron];
        }

        delta *= deltaSum;
        delta *= n.GetActivationFunction()(us[layer][neuron], true);

        // Weights
        for (size_t weight = 0; weight < n.GetWeightsReference().size(); weight++) {
          n.GetWeightsReference()[weight] -= (as[layer - 1][weight] * delta) * _learningRate;
        }

        // Bias
        n.GetBiasReference() -= delta * _learningRate;

        ds[layer][neuron] = delta;
      }
      // Other layer and first layer
      else if (layer != _model.GetLayerCount() - 1 && layer == 0) {
        delta = 1;
        deltaSum = 0;
        for (size_t nextLayerNeuron = 0; nextLayerNeuron < _model.GetLayerSize(layer + 1); nextLayerNeuron++) {
          deltaSum += ds[layer + 1][nextLayerNeuron] * _model.GetNeuron(layer + 1, nextLayerNeuron).GetWeightsReference()[neuron];
        }

        delta *= deltaSum;
        delta *= n.GetActivationFunction()(us[layer][neuron], true);

        // Weights
        for (size_t weight = 0; weight < n.GetWeightsReference().size(); weight++) {
          n.GetWeightsReference()[weight] -= (_inputs[weight] * delta) * _learningRate;
        }

        // Bias
        n.GetBiasReference() -= delta * _learningRate;

        ds[layer][neuron] = delta;
      }
    }
  }
}

void Network::TrainSingle(
    Model &_model, const std::vector<std::vector<double_t>> &_batchInputs,
    const std::vector<std::vector<double_t>> &_batchExpectedOutputs,
    const double_t _learningRate) {
  if (_batchInputs.size() != _batchExpectedOutputs.size()) {
    throw std::runtime_error(
        "Input batch size does not equal output batch size");
  }

  for (size_t i = 0; i < _batchInputs.size(); i++) {
    double_t errorStart, errorEnd;
    errorStart = Neuro::Network::CalculateMeanError(_model, _batchInputs[i],
                                                    _batchExpectedOutputs[i]);
    BackwardPropagate(_model, _batchInputs[i], _batchExpectedOutputs[i],
                      _learningRate);
    errorEnd = Neuro::Network::CalculateMeanError(_model, _batchInputs[i],
                                                  _batchExpectedOutputs[i]);
    // std::cout << "From " << errorStart << " to " << errorEnd << std::endl;
  }
}

void Network::Train(
    const size_t _iterations, Model &_model,
    const std::vector<std::vector<double_t>> &_batchInputs,
    const std::vector<std::vector<double_t>> &_batchExpectedOutputs,
    double_t _learningRate) {
  if (!_model.IsValid()) {
    throw std::runtime_error("Model not loaded correctly");
  }

  size_t iteration = 0;
  double_t meanErrorLast, meanError;
  meanErrorLast = CalculateBatchMeanError(_model, _batchInputs, _batchExpectedOutputs);

  while (iteration < _iterations) {
    Model m = _model;

    TrainSingle(m, _batchInputs, _batchExpectedOutputs, _learningRate);
    meanError = CalculateBatchMeanError(_model, _batchInputs, _batchExpectedOutputs);
    _model = m;
    if (iteration % 100 == 0) {
      std::cout << '\r' << "\tBatch Mean Error: " << meanError << '\r' << std::flush;
    }

    if (meanError > meanErrorLast) {
      _learningRate *= 0.99;
    }
    else {
      _learningRate *= 1.00001;
    }

    meanErrorLast = meanError;

    // if (meanError <= meanErrorLast) {
    //   meanErrorLast = meanError;
    //   _model = m;
    //   std::cout << "Batch Mean Error: " << meanError << std::endl;
    // } else {
    //   _learningRate *= 0.99;
    // }

    if (_learningRate <= 0.00000000001) {
      break;
    }

    iteration++;
  }

  std::cout << std::endl;

  std::cout << "Batches Trained: " << iteration << std::endl;
}

} // namespace Neuro
