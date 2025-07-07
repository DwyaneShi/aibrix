#pragma once

namespace aibrix {
namespace internal {
class no_assign {
public:
  void operator=(const no_assign &) = delete;
  no_assign(const no_assign &) = default;
  no_assign() = default;
};

//! Base class for types that should not be copied or assigned.
class no_copy : no_assign {
public:
  no_copy(const no_copy &) = delete;
  no_copy() = default;
};
} // namespace internal
} // namespace aibrix
