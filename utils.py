from typing import Optional


import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


class MedianPool2d(nn.Module):

    def __init__(self, kernel_size=3, stride=1, padding=0, same=True):
        super().__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


def normalization(x, mean: Optional[torch.Tensor]=None, std: Optional[torch.Tensor]=None):
    if mean is None:
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    if std is None:
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    return (x - mean) / (std + 1e-6)


def denormalization(x, mean: Optional[torch.Tensor]=None, std: Optional[torch.Tensor]=None):
    if mean is None:
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    if std is None:
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    return x * std + mean

def convert_relu_to_sigmoid(model):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(model, child_name, torch.nn.Sigmoid())
        else:
            convert_relu_to_sigmoid(child)

def upscale(imgs):
    return torch.nn.functional.interpolate(imgs, scale_factor=2)


def get_resnet18(is_sigmoid=True):
    model = torchvision.models.resnet18(pretrained=False)
    if is_sigmoid:
        convert_relu_to_sigmoid(model)
    return model