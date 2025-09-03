#pragma once
#include <cmath>
#include <map>
#include <stdexcept>

namespace Neuro {
typedef double_t (*ACTIVATION_FUNCTION)(double_t _input, bool _derivative);
typedef double_t (*ERROR_FUNCTION)(double_t _output, double_t _expected, bool _derivative);

inline double_t A_NONE(double_t _input, bool _derivative = false) {
  if (_derivative) {
    return 1;
  } else {
    return _input;
  }
}

inline double_t A_RELU(double_t _input, bool _derivative = false) {
  if (_derivative) {
    if (_input == 0) {
      throw std::invalid_argument("Divide by zero error");
    }

    return (_input >= 0) ? 1 : 0;
  } else {
    return (_input > 0) ? _input : 0;
  }
}

inline double_t A_SIGMOID(double_t _input, bool _derivative = false) {
  if (_derivative) {
    return A_SIGMOID(_input) * (1 - A_SIGMOID(_input));
  } else {
    return 1 / (1 + std::exp(-_input));
  }
}

inline double_t A_TANH(double_t _input, bool _derivative = false) {
  if (_derivative) {
    return std::pow(2 / (std::exp(_input) + std::exp(-_input)), 2);
  } else {
    return (std::exp(_input) - std::exp(-_input)) /
           (std::exp(_input) + std::exp(-_input));
  }
}

inline double_t C_ERROR(double_t _output, double_t _expected, bool _derivative = false) {
  if (_derivative) {
    return 1;
  } else {
    return (_output - _expected);
  }
}

inline double_t C_INVERSE_ERROR(double_t _output, double_t _expected, bool _derivative = false) {
  if (_derivative) {
    return -1;
  } else {
    return (_expected - _output);
  }
}

inline double_t C_SQUARED_ERROR(double_t _output, double_t _expected, bool _derivative = false) {
  if (_derivative) {
    return C_ERROR(_output, _expected, _derivative) * 2 * C_ERROR(_output, _expected, false);
  } else {
    return std::pow(C_ERROR(_output, _expected, _derivative), 2);
  }
}

inline double_t C_INVERSE_SQUARED_ERROR(double_t _output, double_t _expected, bool _derivative = false) {
  if (_derivative) {
    return C_INVERSE_ERROR(_output, _expected, _derivative) * 2 * C_INVERSE_ERROR(_output, _expected, false);
  } else {
    return std::pow(C_INVERSE_ERROR(_output, _expected, _derivative), 2);
  }
}

inline double_t C_ABSOLUTE_ERROR(double_t _output, double_t _expected, bool _derivative = false) {
  if (_derivative) {
    if (_output < _expected) {
      return -1;
    }
    else {
      return 1;
    }
  } else {
    return std::abs(C_ERROR(_output, _expected, _derivative));
  }
}

static const std::map<std::string, ACTIVATION_FUNCTION> ACTIVATION_FUNCTIONS_LIST = {
  {"NONE", A_NONE},
  {"RELU", A_RELU},
  {"SIGMOID", A_SIGMOID},
  {"TANH", A_TANH}
};

static const std::map<std::string, ERROR_FUNCTION> ERROR_FUNCTION_LIST = {
  {"ERROR", C_ERROR},
  {"INVERSE_ERROR", C_INVERSE_ERROR},
  {"SQUARED_ERROR", C_SQUARED_ERROR},
  {"INVERSE_SQUARED_ERROR", C_INVERSE_SQUARED_ERROR},
  {"ABSOLUTE_ERROR", C_ABSOLUTE_ERROR}
};
} // namespace Neuro
