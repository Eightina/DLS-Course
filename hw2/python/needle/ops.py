"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List, Tuple
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(array_api.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad, out_grad)


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad,)


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return (out_grad * rhs, out_grad * lhs)


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        (ipt,) = node.inputs
        return (out_grad * self.scalar * ipt ** (self.scalar - 1),)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a: NDArray, b: NDArray):
        return a / b

    def gradient(self, out_grad: Tensor, node: Tensor):
        l, r = node.inputs
        return (out_grad / r, -out_grad * l / r ** (2))
        # return (
        #     multiply(out_grad, power_scalar(r, -1)),
        #     multiply(multiply(mul_scalar(out_grad, -1), l), power_scalar(r, -2)),
        # )


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a / self.scalar

    def gradient(self, out_grad: Tensor, node):
        return (out_grad / self.scalar,)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray):
        if self.axes != None:
            ax1, ax2 = self.axes
            return array_api.swapaxes(a, ax1, ax2)
        return array_api.swapaxes(a, a.ndim - 2, a.ndim - 1)

    def gradient(self, out_grad: Tensor, node: Tensor):
        if self.axes == None:
            self.axes = (out_grad.ndim - 2, out_grad.ndim - 1)
        return (out_grad.transpose(axes=self.axes),)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad.reshape(node.inputs[0].shape),)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        ipt_shape = list(node.inputs[0].shape)
        axes = []
        ipt_shape = [1] * (len(self.shape) - len(ipt_shape)) + ipt_shape
        for i, s in enumerate(self.shape):
            if i >= len(ipt_shape) or s != ipt_shape[i]:
                axes.append(i)
        return (reshape(summation(out_grad, tuple(axes)), ipt_shape),)

        # lhs = node.inputs[0]  # 算子的输入，也就是b=[3]
        # origin_shape = lhs.shape  # 原本的形状，也就是b.shape=(1,)
        # target_shape = self.shape  # 想要变换到的形状，也就是b_exp.shape=(1,2)
        # expanded_axes = []  # 记录哪一个维度被拓展了
        # for i in range(-1, -len(target_shape)-1, -1):  # 从尾部开始遍历
        #     if i < -len(origin_shape):
        #         # origin_shape的长度可能会比target_shape短，
        #         # 比如origin_shape=(1,)，target_shape=(1,2)。
        #         expanded_axes.append(i+len(target_shape))
        #         continue
        #     if target_shape[i] != origin_shape[i]:
        #         # 如果目标形状与原本的形状不相同
        #         # 那就说明这个维度经过了拓展，需要记录到expanded_axes中
        #         expanded_axes.append(i + len(target_shape))
        # # out_grad进行sum运算，运算的轴axes是b_exp相对于b经过拓展的维度
        # res = summation(out_grad, tuple(expanded_axes))
        # # 因为res的形状可能与lhs(也就是b)不相同，所以这里需要reshape到b原本的维度上。
        # res = reshape(res, origin_shape)
        # return res

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[Tuple[int]] = None):
        self.axes = axes

    def compute(self, a: NDArray):
        # if self.axes != None:
            # ax, = self.axes
        # return array_api.sum(a)
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor):
        # print(out_grad.shape)
        # print(node.inputs[0].shape)
        new_shape = list(out_grad.shape)
        if self.axes != None:
            for i in self.axes:
                new_shape.insert(i, 1)
        out_grad = reshape(out_grad, new_shape)
        return (broadcast_to(out_grad, node.inputs[0].shape),)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return array_api.matmul(a, b)

    def gradient(self, out_grad: Tensor, node: Tensor):
        l: Tensor
        r: Tensor
        l_grad: Tensor
        r_grad: Tensor

        (l, r) = node.inputs
        l_grad = out_grad @ r.transpose()
        r_grad = l.transpose() @ out_grad

        if l_grad.shape != l.shape:
            l_grad = l_grad.sum(
                tuple(range(l_grad.ndim - l.ndim))
            )  # doing summation on the dims representing batches

        if r_grad.shape != r.shape:
            r_grad = r_grad.sum(tuple(range(r_grad.ndim - r.ndim)))

        return (l_grad, r_grad)
        # caution! eighter of l or r can be batched like 6 * 6 * 5 * 4, 4 * 3


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a: NDArray):
        return array_api.negative(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (-out_grad,)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        (ipt,) = node.inputs
        return (out_grad / ipt,)


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a: NDArray):
        return array_api.exp(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (exp(node.inputs[0]) * out_grad,)


def exp(a):
    return Exp()(a)


class Binary(TensorOp):
    def compute(self, a: NDArray):
        return a.astype(array_api.bool8).astype(array_api.float32)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (
            Tensor(
                [
                    [0.0 for i in range(node.inputs[0].shape[1])]
                    for j in range(node.inputs[0].shape[0])
                ]
            ),
        )


def binary(a):
    return Binary()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return (array_api.abs(a) + a) / 2

    def gradient(self, out_grad: Tensor, node: Tensor):
        # return (
        #     Tensor(array_api.sign(node.realize_cached_data())),
        # )
        a = node.inputs[0].realize_cached_data()
        mask = Tensor(a > 0, requires_grad=False)
        return (out_grad * mask,)

def relu(a):
    return ReLU()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
