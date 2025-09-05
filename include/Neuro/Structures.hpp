#pragma once

#include <cmath>
#include <cstddef>
#include <vector>
#include <mutex>

namespace Neuro {
class StructureLayer {
private:
  mutable std::mutex m_accessing;
  std::vector<double_t> m_values;

public:
  StructureLayer() = default;
  StructureLayer(StructureLayer&& _obj);
  StructureLayer(const StructureLayer& _obj);
  StructureLayer& operator=(StructureLayer&& _obj);
  StructureLayer& operator=(const StructureLayer& _obj);

  double_t &operator[](const size_t _neuron);
  const double_t &operator[](const size_t _neuron) const;

  const size_t size() const;
  void resize(const size_t _newSize);
};


class Structure {
protected:
  Structure() = default;

private:
  mutable std::mutex m_accessing;
  std::vector<StructureLayer> m_values;

public:
  Structure(Structure&& _obj);
  Structure(const Structure& _obj);
  Structure& operator=(Structure&& _obj);
  Structure& operator=(const Structure& _obj);

  StructureLayer &operator[](const size_t _layer);
  const StructureLayer &operator[](const size_t _layer) const;

  const size_t size() const;
  void resize(const size_t _newSize);
};


class ActivatedStructure : public Structure {
public:
  ActivatedStructure();
};

class UnactivatedStructure : public Structure {
public:
  UnactivatedStructure();
};

class DeltaStructure : public Structure {
public:
  DeltaStructure();
};

} // namespace Neuro
