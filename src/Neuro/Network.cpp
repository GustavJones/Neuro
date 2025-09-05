#include "Neuro/Network.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <future>
#include <chrono>

namespace Neuro {
static const bool USE_THREADS = true;

void Network::CalculateStructures(const Model &_model, const std::vector<double_t> &_inputs, UnactivatedStructure &_unactivated, ActivatedStructure &_activated) {
  const bool MULTITHREADED = false;

  std::vector<std::future<void>> threads;
  _unactivated.resize(_model.GetLayerCount());
  _activated.resize(_model.GetLayerCount());

  for (size_t layer = 0; layer < _model.GetLayerCount(); layer++) {
    _unactivated[layer].resize(_model.GetLayerSize(layer));
    _activated[layer].resize(_model.GetLayerSize(layer));
  }

  std::mutex m;
  std::vector<double_t> previousLayerOutputs = _inputs;
  std::vector<double_t> layerOutputs;

  for (size_t layer = 0; layer < _model.GetLayerCount(); layer++) {
    layerOutputs.resize(_model.GetLayerSize(layer));
    threads.clear();

    if (MULTITHREADED)
    {
      threads.reserve(_model.GetLayerSize(layer));
      for (size_t neuron = 0; neuron < _model.GetLayerSize(layer); neuron++) {
        threads.push_back(
          std::async(std::launch::async,
            [&_model, &_unactivated, &_activated, layer, neuron, &m, &previousLayerOutputs, &layerOutputs]() {
              double_t unactivatedOutput = _model.GetNeuron(layer, neuron).Calculate(previousLayerOutputs, false);
              double_t activatedOutput = _model.GetNeuron(layer, neuron).GetActivationFunction()(unactivatedOutput, false);

              m.lock();
              _unactivated[layer][neuron] = unactivatedOutput;
              _activated[layer][neuron] = activatedOutput;
              layerOutputs[neuron] = _activated[layer][neuron];
              m.unlock();
            }
          )
        );
      }

      for (const auto& thread : threads)
      {
        thread.wait();
      }
    }
    else
    {
      for (size_t neuron = 0; neuron < _model.GetLayerSize(layer); neuron++) {
        double_t unactivatedOutput = _model.GetNeuron(layer, neuron).Calculate(previousLayerOutputs, false);
        double_t activatedOutput = _model.GetNeuron(layer, neuron).GetActivationFunction()(unactivatedOutput, false);

        m.lock();
        _unactivated[layer][neuron] = unactivatedOutput;
        _activated[layer][neuron] = activatedOutput;
        layerOutputs[neuron] = _activated[layer][neuron];
        m.unlock();
      }
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
  const bool MULTITHREADED = false;

  std::vector<std::future<double_t>> threads;
  threads.reserve(_batchInputs.size());
  double_t total = 0;

  for (size_t i = 0; i < _batchInputs.size(); i++) {
    if (MULTITHREADED)
    {
      threads.push_back(std::async(std::launch::async, &CalculateMeanError, _model, _batchInputs[i], _batchExpectedOutputs[i]));
    }
    else
    {
      total += CalculateMeanError(_model, _batchInputs[i], _batchExpectedOutputs[i]);
    }
  }

  if (MULTITHREADED)
  {
    for (auto& thread : threads)
    {
      total += thread.get();
    }
  }

  total /= _batchInputs.size();
  return total;
}

std::vector<double_t> Network::Calculate(const Model &_model,
                                        const std::vector<double_t> &_inputs) {
  std::vector<double_t> output;

  if (!_model.IsValid()) {
    throw std::runtime_error("Model not loaded correctly");
  }

  UnactivatedStructure us;
  ActivatedStructure as;

  CalculateStructures(_model, _inputs, us, as);

  for (size_t i = 0; i < as[_model.GetLayerCount() - 1].size(); i++)
  {
    output.push_back(as[_model.GetLayerCount() - 1][i]);
  }

  return output;
}

void Network::BackwardPropagate(Model &_model,
                                const std::vector<double_t> &_inputs,
                                const std::vector<double_t> &_expected,
                                const double_t _learningRate) {
  const bool MULTITHREADED = false;
  std::vector<std::future<void>> threads;

  DeltaStructure ds;

  UnactivatedStructure us;
  ActivatedStructure as;
  CalculateStructures(_model, _inputs, us, as);

  ds.resize(_model.GetLayerCount());
  for (size_t layer = 0; layer < _model.GetLayerCount(); layer++) {
    ds[layer].resize(_model.GetLayerSize(layer));
  }

  auto threadLambda = [&_model, &as, &us, &ds, &_expected, &_inputs, &_learningRate](const size_t layer, const size_t neuron) {
    double_t delta, deltaSum;
    Neuro::Neuron &n = _model.GetNeuron(layer, neuron);

    // Last Layer and not first layer
    if (layer == _model.GetLayerCount() - 1 && layer > 0) {
      delta = 1;
      delta /= _model.GetLayerSize(_model.GetLayerCount() - 1);
      delta *= _model.GetErrorFunction()(as[layer][neuron], _expected[neuron], true);
      delta *= n.GetActivationFunction()(us[layer][neuron], true);

      // Weights
      for (size_t weight = 0; weight < n.GetWeightsAmount(); weight++) {
        n.SetWeight(weight, n.GetWeight(weight) - (as[layer - 1][weight] * delta) * _learningRate);
      }

      // Bias
      n.SetBias(n.GetBias() - delta * _learningRate);

      ds[layer][neuron] = delta;
    }
    // Last layer and first layer
    else if (layer == _model.GetLayerCount() - 1 && layer == 0) {
      delta = 1;
      delta /= _model.GetLayerSize(_model.GetLayerCount() - 1);
      delta *= _model.GetErrorFunction()(as[layer][neuron], _expected[neuron], true);
      delta *= n.GetActivationFunction()(us[layer][neuron], true);

      // Weights
      for (size_t weight = 0; weight < n.GetWeightsAmount(); weight++) {
        n.SetWeight(weight, n.GetWeight(weight) - (_inputs[weight] * delta) * _learningRate);
      }

      // Bias
      n.SetBias(n.GetBias() - delta * _learningRate);

      ds[layer][neuron] = delta;
    }
    // Other layer and not first layer
    else if (layer != _model.GetLayerCount() - 1 && layer > 0) {
      delta = 1;
      deltaSum = 0;
      for (size_t nextLayerNeuron = 0; nextLayerNeuron < _model.GetLayerSize(layer + 1); nextLayerNeuron++) {
        deltaSum += ds[layer + 1][nextLayerNeuron] * _model.GetNeuron(layer + 1, nextLayerNeuron).GetWeight(neuron);
      }

      delta *= deltaSum;
      delta *= n.GetActivationFunction()(us[layer][neuron], true);

      // Weights
      for (size_t weight = 0; weight < n.GetWeightsAmount(); weight++) {
        n.SetWeight(weight, n.GetWeight(weight) - (as[layer - 1][weight] * delta) * _learningRate);
      }

      // Bias
      n.SetBias(n.GetBias() - delta * _learningRate);

      ds[layer][neuron] = delta;
    }
    // Other layer and first layer
    else if (layer != _model.GetLayerCount() - 1 && layer == 0) {
      delta = 1;
      deltaSum = 0;
      for (size_t nextLayerNeuron = 0; nextLayerNeuron < _model.GetLayerSize(layer + 1); nextLayerNeuron++) {
        deltaSum += ds[layer + 1][nextLayerNeuron] * _model.GetNeuron(layer + 1, nextLayerNeuron).GetWeight(neuron);
      }

      delta *= deltaSum;
      delta *= n.GetActivationFunction()(us[layer][neuron], true);

      // Weights
      for (size_t weight = 0; weight < n.GetWeightsAmount(); weight++) {
        n.SetWeight(weight, n.GetWeight(weight) - (_inputs[weight] * delta) * _learningRate);
      }

      // Bias
      n.SetBias(n.GetBias() - delta * _learningRate);

      ds[layer][neuron] = delta;
    }
  };

  for (int64_t layer = _model.GetLayerCount() - 1; layer >= 0; layer--) {
    threads.clear();
    threads.reserve(_model.GetLayerSize(layer));

    for (size_t neuron = 0; neuron < _model.GetLayerSize(layer); neuron++) {
      if (MULTITHREADED)
      {
        threads.push_back(std::async(std::launch::async, threadLambda, layer, neuron));
      }
      else
      {
        threadLambda(layer, neuron);
      }
    }

    if (MULTITHREADED)
    {
      for (const auto& thread : threads)
      {
        thread.wait();
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

  std::chrono::time_point<std::chrono::steady_clock> timerLast, timerCurrent;

  timerLast = std::chrono::steady_clock::now();

  while (iteration < _iterations) {
    Model m = _model;
    
    auto first = std::chrono::high_resolution_clock::now();
    TrainSingle(m, _batchInputs, _batchExpectedOutputs, _learningRate);
    auto last = std::chrono::high_resolution_clock::now();

    meanError = CalculateBatchMeanError(_model, _batchInputs, _batchExpectedOutputs);
    _model = m;

    timerCurrent = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(timerCurrent - timerLast).count() >= 200)
    {
      std::cout << '\r' << "\x1b[0K" << "\tBatch Mean Error: " << meanError << "\tTrain Time: " << std::chrono::duration_cast<std::chrono::microseconds>(last - first).count() << " micro seconds" << '\r' << std::flush;
      timerLast = timerCurrent;
    }

    if (meanError > meanErrorLast) {
      _learningRate *= 0.99;
    }
    else {
      _learningRate *= 1.00001;
    }

    meanErrorLast = meanError;

    if (_learningRate <= 0.00000000001) {
      break;
    }

    iteration++;
  }

  std::cout << std::endl;

  std::cout << "Batches Trained: " << iteration << std::endl;
}

} // namespace Neuro
