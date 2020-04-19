[![PyPI](https://img.shields.io/pypi/v/rfconv.svg)](https://pypi.python.org/pypi/rfconv)
[![PyPI Pre-release](https://img.shields.io/badge/pypi--prerelease-v0.0.1-ff69b4.svg)](https://pypi.org/project/rfconv/#history)
[![pytest](https://github.com/zhanghang1989/RFConv/workflows/pytest/badge.svg)](https://github.com/zhanghang1989/RFConv/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# RFConv
Rectified Convolution


## Pretrained Model

| model            | baseline | rectified |
|------------------|----------|-----------|
| ResNet-50        | 76.66    | 77.10     |
| ResNet-101       | 78.13    | 78.74     |
| ResNeXt-50_32x4d | 78.17    | 78.48     |


## Verify Models:

**Note:** the inference speed reported in the paper are tested using Gluon implementation with RecordIO data.

### Prepare ImageNet dataset:

Here we use raw image data format for simplicity, please follow [GluonCV tutorial](https://gluon-cv.mxnet.io/build/examples_datasets/recordio.html) if you would like to use RecordIO format.

```bash
cd scripts/dataset/
# assuming you have downloaded the dataset in the current folder
python prepare_imagenet.py --download-dir ./
```

### Test Model

```bash
# use resnest50 as an example
cd scripts/
python verify.py --model resnet50 --crop-size 224
```