from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

__all__ = ["svcca_distance", "pwcca_distance", "CCAHook"]

_device = "cuda" if torch.cuda.is_available() else "cpu"


def svd_reduction(tensor: torch.Tensor, accept_rate=0.99):
    left, diag, right = torch.svd(tensor)
    full = diag.abs().sum()
    ratio = diag.abs().cumsum(dim=0) / full
    num = torch.where(ratio < accept_rate,
                      torch.ones(1).to(ratio.device),
                      torch.zeros(1).to(ratio.device)
                      ).sum()
    return tensor @ right[:, :int(num)]


def zero_mean(tensor: torch.Tensor, dim):
    return tensor - tensor.mean(dim=dim, keepdim=True)


def _svd_cca(x, y):
    u_1, s_1, v_1 = x.svd()
    u_2, s_2, v_2 = y.svd()
    uu = u_1.t() @ u_2
    try:
        u, diag, v = (uu).svd()
    except RuntimeError as e:
        raise e
    a = v_1 @ s_1.reciprocal().diag() @ u
    b = v_2 @ s_2.reciprocal().diag() @ v
    return a, b, diag


def _cca(x, y, method):
    SVCCA distance proposed in Raghu et al. 2017
    :param x: data matrix [data, neurons]
    :param y: data matrix [data, neurons]
    :param method: computational method "svd" (default) or "qr"
    Project Weighting CCA proposed in Marcos et al. 2018
    :param x: data matrix [data, neurons]
    :param y: data matrix [data, neurons]
    :param method: computational method "svd" (default) or "qr"


class CCAHook(object):
    _supported_modules = (nn.Conv2d, nn.Linear)
    _cca_distance_function = {"svcca": svcca_distance,
                              "pwcca": pwcca_distance}

    def __init__(self,
                 model: nn.Module,
                 name: str, *,
                 cca_distance: str or Callable = pwcca_distance,
                 svd_device: str or torch.device = _device):

        self.model = model
        self.name = name
        _dict = {n: m for n, m in self.model.named_modules()}
        if self.name not in _dict.keys():
            raise NameError(f"No such name ({self.name}) in the model")
        if type(_dict[self.name]) not in self._supported_modules:
            raise TypeError(f"{type(_dict[self.name])} is not supported")

        self._module = _dict[self.name]
        self._module = {n: m for n, m in self.model.named_modules()}[self.name]
        self._hooked_value = None
        self._register_hook()
        if type(cca_distance) == str:
            cca_distance = self._cca_distance_function[cca_distance]
        self._cca_distance = cca_distance

        if svd_device not in ("cpu", "cuda"):
            raise RuntimeError(f"Unknown device name {svd_device}")

        if svd_device == "cpu":
            from multiprocessing import cpu_count

            torch.set_num_threads(cpu_count())
        self._svd_device = svd_device

    def clear(self):
        self._hooked_value = None

    def distance(self, other, size: int or tuple = None):
        tensor1 = self.get_hooked_value()
        tensor2 = other.get_hooked_value()
        if tensor1.dim() != tensor2.dim():
            raise RuntimeError("tensor dimensions are incompatible!")
        tensor1 = tensor1.to(self._svd_device)
        tensor2 = tensor2.to(self._svd_device)
        if isinstance(self._module, nn.Linear):
            return self._cca_distance(tensor1, tensor2).item()
        elif isinstance(self._module, nn.Conv2d):
            return CCAHook._conv2d(tensor1, tensor2, self._cca_distance, size).item()
    def _register_hook(self):

        def hook(_, __, output):
            self._hooked_value = output

        self._module.register_forward_hook(hook)

    def get_hooked_value(self):
        if self._hooked_value is None:
            raise RuntimeError("Please do model.forward() before CCA!")
        return self._hooked_value

    @staticmethod
    def _conv2d_reshape(tensor, size):
        b, c, h, w = tensor.shape
        if size is not None:
            if (size, size) > (h, w):
                raise RuntimeError(f"`size` should be smaller than the tensor's size but ({h}, {w})")
            tensor = F.adaptive_avg_pool2d(tensor, size)
        tensor = tensor.reshape(b, c, -1).permute(2, 0, 1)
        return tensor

    @staticmethod
    def _conv2d(tensor1, tensor2, cca_function, size):
        if tensor1.shape != tensor2.shape:
            raise RuntimeError("tensors' shapes are incompatible!")
        tensor1 = CCAHook._conv2d_reshape(tensor1, size)
        tensor2 = CCAHook._conv2d_reshape(tensor2, size)
        return torch.Tensor([cca_function(t1, t2)
                             for t1, t2 in zip(tensor1, tensor2)]).mean()

    @staticmethod
    def data(dataset: Dataset, batch_size: int, *, num_workers: int = 2):
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        input, _ = next(iter(data_loader))
        return input
