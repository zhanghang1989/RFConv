#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

void CONV_RECTIFY_CUDA(
  at::Tensor& output,
  const at::Tensor& input,
  at::IntArrayRef kernel_size,
  at::IntArrayRef stride,
  at::IntArrayRef padding,
  at::IntArrayRef dilation,
  bool avg_mode);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv_rectify", &CONV_RECTIFY_CUDA, "Convolution Rectifier (CUDA)");
}
