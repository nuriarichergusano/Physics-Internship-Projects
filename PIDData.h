#pragma once

#include <torch/types.h>
#include <torch/data/example.h>

namespace torch {
namespace data {

/// An `PIDData` from a dataset.
///
/// A dataset consists of data and an associated target (label).
template <typename Data = at::Tensor, typename Target = at::Tensor, typename Energies = at::Tensor>
struct PIDData : public torch::data::Example<Data, Target> {
  //using DataType = Data;
  //using TargetType = Target;
  using EnergyType = Energies;

  PIDData() = default;
  PIDData(Data data, Target target, Energies energies)
      : torch::data::Example<Data, Target>(data, target), energies(std::move(energies)) {}

  Energies energies;
};

namespace example {
using NoTarget = void;
} // namespace example


/// An `PIDData` from a dataset.
///
/// A dataset consists of data and an associated target (label).
#ifdef DIS
template <typename Data = at::Tensor, typename Target = at::Tensor>
struct PIDData<Data, Target, example::NoTarget> {
  using DataType = Data;
  using TargetType = Target;

  PIDData() = default;
  PIDData(Data data, Target target)
      : data(std::move(data)), target(std::move(target)) {}

  operator Data&() {
    return data;
  }
  operator const Data&() const {
    return data;
  }
    operator Target&() {
    return target;
  }
  operator const Target&() const {
    return target;
  }

  Data data;
  Target target;
};
#endif

/// A specialization for `PIDData` that does not have a target.
///
/// This class exists so that code can be written for a templated `PIDData`
/// type, and work both for labeled and unlabeled datasets.
template <typename Data>
struct PIDData<Data, example::NoTarget, example::NoTarget> {
  using DataType = Data;
  using TargetType = example::NoTarget;

  PIDData() = default;
  /* implicit */ PIDData(Data data) : data(std::move(data)) {}

  // When a DataLoader returns an PIDData like this, that example should be
  // implicitly convertible to the underlying data type.

  operator Data&() {
    return data;
  }
  operator const Data&() const {
    return data;
  }

  Data data;
};

using TensorPIDData = PIDData<at::Tensor, example::NoTarget>;

} // namespace data
} // namespace torch
