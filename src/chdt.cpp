#include <torch/extension.h>
#include "const.h"

TORCH_LIBRARY(chdt, m) {
  m.def("transform(Tensor input) -> Tensor");
}

torch::Tensor chdt_cpu(const torch::Tensor &input);
TORCH_LIBRARY_IMPL(chdt, CPU, m) {
    m.impl("transform", chdt_cpu);
}

torch::Tensor transform(const torch::Tensor &input) {
  static auto op = torch::Dispatcher::singleton()
    .findSchemaOrThrow("chdt::transform", "")
    .typed<decltype(transform)>();
  return op.call(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transform", &transform, "Chebyshev distance transform");
    m.attr("INF") = py::int_(INF);
}

