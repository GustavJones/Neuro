#include "Neuro/Structures.hpp"

namespace Neuro {
double_t &StructureLayer::operator[](const size_t _neuron) {
  return m_values[_neuron];
}

const double_t &StructureLayer::operator[](const size_t _neuron) const {
  return m_values[_neuron];
}

std::vector<double_t> &StructureLayer::GetValues() { return m_values; }

const std::vector<double_t> &StructureLayer::GetValues() const {
  return m_values;
}

StructureLayer &Structure::operator[](const size_t _layer) {
  return m_values[_layer];
}

const StructureLayer &Structure::operator[](const size_t _layer) const {
  return m_values[_layer];
}

std::vector<StructureLayer> &Structure::GetValues() { return m_values; }
const std::vector<StructureLayer> &Structure::GetValues() const {
  return m_values;
}

ActivatedStructure::ActivatedStructure() : Structure() {}
UnactivatedStructure::UnactivatedStructure() : Structure() {}
DeltaStructure::DeltaStructure() {}
} // namespace Neuro
