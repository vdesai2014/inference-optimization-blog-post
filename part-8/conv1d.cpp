#include <torch/extension.h>

torch::Tensor conv1d_fwd(
    torch::Tensor& input,
    torch::Tensor& conv1d_weight_tensor,
    torch::Tensor& conv1d_bias_tensor,
    int output_channels,
    int padding,
    int kernel_size
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d", &conv1d_fwd, "1D convolution wrapper function",
          py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("output_channels"), py::arg("padding"), py::arg("kernel_size"));
}