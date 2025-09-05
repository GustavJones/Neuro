#include "Neuro/Structures.hpp"
#include <thread>
#include <chrono>

namespace Neuro {
StructureLayer::StructureLayer(StructureLayer&& _obj) {
  m_accessing.lock();

  resize(_obj.size());

  for (size_t index = 0; index < _obj.size(); index++)
  {
    operator[](index) = _obj[index];
  }

  m_accessing.unlock();
}

StructureLayer::StructureLayer(const StructureLayer& _obj) {
  m_accessing.lock();

  resize(_obj.size());

  for (size_t index = 0; index < _obj.size(); index++)
  {
    operator[](index) = _obj[index];
  }

  m_accessing.unlock();
}

StructureLayer& StructureLayer::operator=(StructureLayer&& _obj) {
  m_accessing.lock();

  resize(_obj.size());

  for (size_t index = 0; index < _obj.size(); index++)
  {
    operator[](index) = _obj[index];
  }

  m_accessing.unlock();

  return *this;
}

StructureLayer& StructureLayer::operator=(const StructureLayer& _obj) {
  m_accessing.lock();

  resize(_obj.size());

  for (size_t index = 0; index < _obj.size(); index++)
  {
    operator[](index) = _obj[index];
  }

  m_accessing.unlock();

  return *this;
}

double_t &StructureLayer::operator[](const size_t _neuron) {
  m_accessing.lock();
  auto &out = m_values[_neuron];
  m_accessing.unlock();

  return out;
}

const double_t &StructureLayer::operator[](const size_t _neuron) const {
  m_accessing.lock();
  auto &out = m_values[_neuron];
  m_accessing.unlock();

  return out;
}

const size_t StructureLayer::size() const {
  m_accessing.lock();
  auto out = m_values.size();
  m_accessing.unlock();

  return out;
}

void StructureLayer::resize(const size_t _newSize) {
  m_accessing.lock();
  m_values.resize(_newSize);
  m_accessing.unlock();
}

Structure::Structure(Structure&& _obj) {
  m_accessing.lock();

  resize(_obj.size());

  for (size_t index = 0; index < _obj.size(); index++)
  {
    operator[](index) = _obj[index];
  }

  m_accessing.unlock();
}

Structure::Structure(const Structure& _obj) {
  m_accessing.lock();

  resize(_obj.size());

  for (size_t index = 0; index < _obj.size(); index++)
  {
    operator[](index) = _obj[index];
  }

  m_accessing.unlock();
}

Structure& Structure::operator=(Structure&& _obj) {
  m_accessing.lock();

  resize(_obj.size());

  for (size_t index = 0; index < _obj.size(); index++)
  {
    operator[](index) = _obj[index];
  }

  m_accessing.unlock();
  return *this;
}

Structure& Structure::operator=(const Structure& _obj) {
  m_accessing.lock();

  resize(_obj.size());

  for (size_t index = 0; index < _obj.size(); index++)
  {
    operator[](index) = _obj[index];
  }

  m_accessing.unlock();
  return *this;
}

StructureLayer &Structure::operator[](const size_t _layer) {
  m_accessing.lock();
  auto &out = m_values[_layer];
  m_accessing.unlock();
  return out;
}

const StructureLayer &Structure::operator[](const size_t _layer) const {
  m_accessing.lock();
  auto &out = m_values[_layer];
  m_accessing.unlock();
  return out;
}

const size_t Structure::size() const {
  m_accessing.lock();
  auto out = m_values.size();
  m_accessing.unlock();

  return out;
}

void Structure::resize(const size_t _newSize) {
  m_accessing.lock();
  m_values.resize(_newSize);
  m_accessing.unlock();
}

ActivatedStructure::ActivatedStructure() : Structure() {}
UnactivatedStructure::UnactivatedStructure() : Structure() {}
DeltaStructure::DeltaStructure() {}
} // namespace Neuro
