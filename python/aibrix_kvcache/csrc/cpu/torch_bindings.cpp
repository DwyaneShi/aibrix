#include "core/registration.h"
#include "tensor_pool.h"

#include <torch/library.h>
#include <torch/version.h>

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _mem_ops), mem_ops) {
  mem_ops.class_<aibrix::tensor_pool>("TensorPool")
      .def(torch::init<const std::vector<std::vector<int64_t>> &, int64_t>())
      .def("allocate", &aibrix::tensor_pool::allocate)
      .def("deallocate", &aibrix::tensor_pool::deallocate);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
