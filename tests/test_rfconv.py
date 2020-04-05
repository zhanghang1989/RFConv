import torch
from rfconv import RFConv2d

def test_RFConv2d():
    x = torch.ones(1, 1, 7, 7)
    for k in [3, 5, 7]:
        pad = k//2
        layer = RFConv2d(1, 1, k, 1, pad)
        y = layer(x)
        print(y)

if __name__ == '__main__':
    test_RFConv2d()
