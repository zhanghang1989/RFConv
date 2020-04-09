#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/div_rtn.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>

#include "utils.h"

template <typename scalar_t, typename accscalar_t>
__global__ void conv_rectify_cuda_frame(
    const int nthreads,
    //const scalar_t* const bottom_data,
    const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    scalar_t* const top_data,
    bool average_mode) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    //const int c = (index / pooled_width / pooled_height) % channels;
    //const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = ((kernel_h - 1) / dilation_h + 1) * ((kernel_w - 1) / dilation_w + 1);
    //const int pool_size = ((hend - hstart - 1) / dilation_h + 1) * ((wend - wstart - 1) / dilation_w + 1);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    accscalar_t mul_factor;
    int hcount = int(((hend - hstart) - 1) / dilation_h + 1);
    int wcount = int(((wend - wstart) - 1) / dilation_w + 1);
    if (average_mode) {
      mul_factor = accscalar_t(1.0) / (hcount * wcount);
    }
    else {
      mul_factor = accscalar_t(1.0) * pool_size / (hcount * wcount);
    }
    top_data[index] = ScalarConvert<accscalar_t, scalar_t>::to(top_data[index] * mul_factor);
  }
}

void conv_rectify_cuda_tempalte(
  at::Tensor& output,
  const at::Tensor& input_,
  at::IntArrayRef kernel_size,
  at::IntArrayRef stride,
  at::IntArrayRef padding,
  at::IntArrayRef dilation,
  bool average)
{
  //at::TensorArg output_arg{ output, "output", 1 };
  //at::TensorArg input_arg{ input_, "input_", 2 };

  //checkAllSameGPU("rectify_out_cuda", {output_arg, input_arg});

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "rectify: kernel_size must either be a single int, or a tuple of two ints");
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
    "rectify: stride must either be omitted, a single int, or a tuple of two ints");
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "rectify: padding must either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
    "rectify: dilation must either be a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  TORCH_CHECK((input_.ndimension() == 3 || input_.ndimension() == 4),
    "non-empty 3D or 4D (batch mode) tensor expected for input");

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

  at::Tensor input = input_.contiguous();

  //output.resize_({nbatch, nInputPlane, outputHeight, outputWidth});

  const int32_t count = safe_downcast<int32_t, int64_t>(output.numel());
  const uint32_t  num_threads = std::min(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);
  const uint32_t num_blocks = at::cuda::ATenCeilDiv<uint32_t>(count, num_threads);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_rectify_cuda_frame", ([&] {
        //using accscalar_t = acc_type<scalar_t, true>;
        scalar_t *output_data = output.data_ptr<scalar_t>();
        conv_rectify_cuda_frame<scalar_t, scalar_t>
            <<<num_blocks, num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
                nbatch,
                nInputPlane,
                inputHeight, inputWidth,
                outputHeight, outputWidth,
                kH, kW,
                dH, dW,
                padH, padW,
                dilationH, dilationW,
                output_data,
                average);
  }));


  AT_CUDA_CHECK(cudaGetLastError());
}

void CONV_RECTIFY_CUDA(
  at::Tensor& output,
  const at::Tensor& input,
  at::IntArrayRef kernel_size,
  at::IntArrayRef stride,
  at::IntArrayRef padding,
  at::IntArrayRef dilation,
  bool average) {
  //at::Tensor output = at::empty({0}, input.options());
  conv_rectify_cuda_tempalte(
    output,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    average);
}


