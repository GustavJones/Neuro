#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace Neuro {
class StructureLayer {
private:
  std::vector<double_t> m_values;

public:
  double_t &operator[](const size_t _neuron);
  const double_t &operator[](const size_t _neuron) const;

  std::vector<double_t> &GetValues();
  const std::vector<double_t> &GetValues() const;
};


class Structure {
protected:
  Structure() = default;

private:
  std::vector<StructureLayer> m_values;

public:
  StructureLayer &operator[](const size_t _layer);
  const StructureLayer &operator[](const size_t _layer) const;

  std::vector<StructureLayer> &GetValues();
  const std::vector<StructureLayer> &GetValues() const;
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
