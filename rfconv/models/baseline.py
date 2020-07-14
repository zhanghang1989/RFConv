##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
from resnest.torch.resnet import ResNet, Bottleneck

__all__ = ['resnet50', 'resnet101', 'resnext50_32x4d']

_url_format = 'https://s3.us-west-1.wasabisys.com/encoding/models/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('8265605f', 'resnet50'),
    ('87134418', 'resnet101'),
    ('3583b05a', 'resnext50_32x4d'),
    ]}

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

rectify_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}

def resnet50(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['radix'] = 0
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            rectify_model_urls['resnet50'], progress=True, check_hash=True,
            map_location=torch.device('cpu')))
    return model

def resnet101(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['radix'] = 0
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            rectify_model_urls['resnet101'], progress=True, check_hash=True,
            map_location=torch.device('cpu')))
    return model

def resnext50_32x4d(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['radix'] = 0
    kwargs['groups'] = 32
    kwargs['bottleneck_width'] = 4
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            rectify_model_urls['resnext50_32x4d'], progress=True, check_hash=True,
            map_location=torch.device('cpu')))
    return model
