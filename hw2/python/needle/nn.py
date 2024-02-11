"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as array_api
from abc import ABC, abstractmethod


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    @abstractmethod
    def forward(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.biased = bias
        self.device = device

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(
            fan_in=self.in_features,
            fan_out=self.out_features,
            device=self.device
        ))

        if not self.biased:
            self.bias = None
            return
        self.bias = Parameter(init.kaiming_uniform(
                fan_in=self.out_features,
                fan_out=1,
                device=self.device
            ).reshape(
                (1, self.out_features)
            ))
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        
        ### BEGIN YOUR SOLUTION
        if not isinstance(self.bias, Tensor):
            return (
                X @ self.weight
            )
            
        return (
            X @ self.weight
            + self.bias.broadcast_to(
                (X.shape[0], self.out_features)
            )
        )
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor):
        ### BEGIN YOUR SOLUTION
        if len(X.shape) <= 1:
            return X
        
        shape_res_mul = 1
        for i in range(1, len(X.shape)):
            shape_res_mul *= X.shape[i]
        return ops.reshape(X, (X.shape[0], shape_res_mul))
        ### END YOUR SOLUTION


class ReLU(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module.forward(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        lse = ops.logsumexp(logits, axes=(1,))
        zy = ops.summation(logits * init.one_hot(logits.shape[-1], y, requires_grad=True), axes=(1,)) 
        return ops.summation(lse - zy) / logits.shape[0]
        ### END YOUR SOLUTION

# class BatchNorm1d(Module):
#     def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
#         super().__init__()
#         self.dim = dim
#         self.eps = eps
#         self.momentum = momentum
#         ### BEGIN YOUR SOLUTION
#         self.weight = Parameter(init.ones(self.dim, requires_grad=True))
#         self.bias = Parameter(init.zeros(self.dim, requires_grad=True))
#         self.running_mean = init.zeros(self.dim)
#         self.running_var = init.ones(self.dim)
#         ### END YOUR SOLUTION


#     def forward(self, x: Tensor) -> Tensor:
#         ### BEGIN YOUR SOLUTION
#         batch_size = x.shape[0]
#         mean = x.sum((0, )).reshape((self.dim)) / batch_size
#         # NOTE array with shape (4, ) is considered as a row, so it can be brcsto (2, 4) and cannot be brcsto (4, 2)
#         x_minus_mean = x - mean.broadcast_to(x.shape)
#         var = (x_minus_mean ** 2).sum((0, )).reshape((self.dim)) / batch_size
        
#         if self.training:
#             self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
#             self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data

#             x_std = ((var + self.eps) ** 0.5).broadcast_to(x.shape)
#             x_normed = x_minus_mean / x_std
#             return x_normed * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
#         else:
#             # NOTE no momentum here!
#             x_normed = (x - self.running_mean) / (self.running_var + self.eps) ** 0.5
#             # NOTE testing time also need self.weight and self.bias
#             return x_normed * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
#         ### END YOUR SOLUTION
class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, requires_grad=True))
        self.running_mean = init.zeros(dim, requires_grad=False)
        self.running_var = init.ones(dim, requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batches = x.shape[0]
        broadcasted_b = self.bias.broadcast_to(x.shape)
        broadcasted_w = self.weight.broadcast_to(x.shape)
        if self.training:
            E = x.sum(axes=(0,)).reshape((self.dim)) / batches
            broadcasted_E = E.broadcast_to(x.shape)
            Var = ((x - broadcasted_E) ** 2).sum(axes=(0,)).reshape((self.dim)) / batches
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * E.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * Var.data
            
            x_std = (Var.broadcast_to(x.shape) + self.eps) ** 0.5
            x_normed = (x - broadcasted_E) / x_std
            
        else:
            x_normed = (x - self.running_mean.broadcast_to(x.shape)) / \
                            ((self.running_var + self.eps) ** 0.5).broadcast_to(x.shape)
                            
        return broadcasted_w * x_normed + broadcasted_b
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones((dim), requires_grad=True))
        self.bias = Parameter(init.zeros((dim), requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        # It seems that E and Var still requires grad?? Why? 
        # Because they are calculated from x and interferes computation of y.
        ### BEGIN YOUR SOLUTION
        batches = x.shape[0]
        broadcasted_E = (x.sum(axes=(1,)).reshape((batches, 1)) / self.dim).broadcast_to(x.shape)
        broadcasted_w = self.weight.broadcast_to(x.shape)
        broadcasted_b = self.bias.broadcast_to(x.shape)
        Var = ((x - broadcasted_E) ** 2).sum(axes=(1,)).reshape((batches, 1)) / self.dim
        y = broadcasted_w * ((x - broadcasted_E) / ((Var.broadcast_to(x.shape) + self.eps) ** 0.5)) + broadcasted_b
        return y
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x
        z = init.randb(*(x.shape), p = 1 - self.p)
        z /= 1 - self.p
        return x * z
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn.forward(x) + x
        ### END YOUR SOLUTION
