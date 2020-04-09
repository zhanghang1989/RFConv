#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <tuple>

#include "utils.h"

template <typename scalar_t>
static void conv_rectify_cpu_frame(
          scalar_t *output_data,
          int64_t nbatch,
          int64_t nInputPlane,
          int64_t inputWidth,
          int64_t inputHeight,
          int64_t outputWidth,
          int64_t outputHeight,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          const int dilation_h,
          const int dilation_w,
          bool average_mode) {
  //at::parallel_for(0, nInputPlane, 0, [&](int64_t start, int64_t end) {
  for (int64_t k = 0; k < nInputPlane; k++) {
    int64_t p;
    for(p = 0; p < nbatch; p++)
    {
      int64_t xx, yy;
      /* For all output pixels... */
      scalar_t *ptr_output = output_data + p*nInputPlane*outputWidth*outputHeight + k*outputWidth*outputHeight;
      //int64_t i;

      for(yy = 0; yy < outputHeight; yy++)
      {
        for(xx = 0; xx < outputWidth; xx++)
        {
          /* Compute the mean of the input image... */
          int64_t hstart = yy * dH - padH;
          int64_t wstart = xx * dW - padW;
          int64_t hend = std::min(hstart + kH, inputHeight + padH);
          int64_t wend = std::min(wstart + kW, inputWidth + padW);
          //int pool_size = (hend - hstart) * (wend - wstart);
          int pool_size = ((kH - 1) / dilation_h + 1) * ((kW - 1) / dilation_w + 1);
          hstart = std::max(hstart, (int64_t) 0);
          wstart = std::max(wstart, (int64_t) 0);
          hend = std::min(hend, inputHeight);
          wend = std::min(wend, inputWidth);
          int hcount = int(((hend - hstart) - 1) / dilation_h + 1);
          int wcount = int(((wend - wstart) - 1) / dilation_w + 1);

          scalar_t mul_factor;
          if (average_mode) {
            mul_factor = scalar_t(1.0) / (hcount * wcount);
          }
          else {
            mul_factor = scalar_t(1.0) * pool_size / (hcount * wcount);
          }
          *ptr_output++ *= mul_factor;
        }
      }
    }
  }
  //});
}

void conv_rectify_cpu_tempalte(
          at::Tensor &output,
          const at::Tensor &input_,
          at::IntArrayRef kernel_size,
          at::IntArrayRef stride, 
          at::IntArrayRef padding,
          at::IntArrayRef dilation,
          bool average_mode)
{
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "conv_rectify: kernel_size must either be a single int, or a tuple of two ints");
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
    "conv_rectify: stride must either be omitted, a single int, or a tuple of two ints");
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "conv_rectify: padding must either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
    "rectify: dilation must either be a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  TORCH_CHECK((input_.ndimension() == 3 || input_.ndimension() == 4),
    "non-empty 2D or 3D (batch mode) tensor expected for input");

  /* sizes */
  const int64_t nbatch = input_.ndimension() == 4 ? input_.size(-4) : 1;
  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  //const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, false);
  //const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, false);
  const int64_t outputHeight = output.size(-2);
  const int64_t outputWidth = output.size(-1);

  pool2d_shape_check(
    input_,
    kH, kW, dH, dW, padH, padW, dilationH, dilationW,
    nInputPlane,
    inputHeight, inputWidth,
    outputHeight, outputWidth);

  TORCH_CHECK(output.is_contiguous(), "conv_rectify: output must be contiguous");

  at::Tensor input = input_.contiguous();

  AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_rectify_cuda_frame", ([&] {
      scalar_t *output_data = output.data_ptr<scalar_t>();
      conv_rectify_cpu_frame<scalar_t>(
        output_data,
        nbatch,
        nInputPlane,
        inputWidth, inputHeight,
        outputWidth, outputHeight,
        kW, kH,
        dW, dH,
        padW, padH,
        dilationH,
        dilationW,
        average_mode);
    }
  ));
}

void CONV_RECTIFY_CPU(
  at::Tensor& output,
  const at::Tensor& input,
  at::IntArrayRef kernel_size,
  at::IntArrayRef stride,
  at::IntArrayRef padding,
  at::IntArrayRef dilation,
  bool average) {
  //at::Tensor output = at::empty({0}, input.options());
  conv_rectify_cpu_tempalte(
    output,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    average);
}

