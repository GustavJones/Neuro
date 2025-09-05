#pragma once
#include "Neuro/Model.hpp"
#include "Neuro/Structures.hpp"
#include <cmath>
#include <vector>

namespace Neuro {
  extern const bool USE_THREADS;

  class Network {
  private:
    Network() = delete;

    static void CalculateStructures(const Model &_model, const std::vector<double_t> &_inputs, UnactivatedStructure &_unactivated, ActivatedStructure &_activated);
    static double_t CalculateMeanError(Model &_model, const std::vector<double_t> &_inputs, const std::vector<double_t> &_expectedOutputs);
    static double_t CalculateBatchMeanError(Model &_model, const std::vector<std::vector<double_t>> &_batchInputs, const std::vector<std::vector<double_t>> &_batchExpectedOutputs);
    static void BackwardPropagate(Model &_model, const std::vector<double_t> &_inputs, const std::vector<double_t> &_expected, const double_t _learningRate);

  public:
    static std::vector<double_t> Calculate(const Model &_model, const std::vector<double_t> &_inputs);
    static void TrainSingle(Model &_model, const std::vector<std::vector<double_t>> &_batchInputs, const std::vector<std::vector<double_t>> &_batchExpectedOutputs, const double_t _learningRate);
    static void Train(const size_t _iterations, Model &_model, const std::vector<std::vector<double_t>> &_batchInputs, const std::vector<std::vector<double_t>> &_batchExpectedOutputs, double_t _learningRate);
  };
}
