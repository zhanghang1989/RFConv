#include <torch/torch.h>
#include <vector>

void CONV_RECTIFY_CPU(
  at::Tensor& output,
  const at::Tensor& input,
  at::IntArrayRef kernel_size,
  at::IntArrayRef stride,
  at::IntArrayRef padding,
  at::IntArrayRef dilation,
  bool average);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv_rectify", &CONV_RECTIFY_CPU, "Convolution Rectifier (CPU)");
}
