from typing import Callable, NamedTuple
import numpy as np

from microtorch.utils import AutogradError, Functions, NoGradError, InplaceOpError


class Dependency(NamedTuple):
    tensor: "Tensor"
    grad_fn: Callable[[np.ndarray], np.ndarray]


class Tensor:
    def __init__(self, data, requires_grad=False, depends_on=None):

        if isinstance(data, Tensor):
            data = data.data
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        self.data = data.astype(np.float)

        self.requires_grad = requires_grad
        self.depends_on = depends_on or list()

        self._grad = None
        if self.requires_grad:
            self.zero_grad()

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    @property
    def shape(self):
        return self.data.shape

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    @property
    def dim(self):
        return self.data.ndim

    def zero_grad(self):
        self._grad = np.zeros_like(self.data)

    @property
    def grad(self):
        if self.requires_grad:
            return self._grad
        raise NoGradError()

    @grad.setter
    def grad(self, value):
        if not self.requires_grad:
            raise NoGradError()
        self._grad = value

    def item(self):
        """In case self is only one number, return this number"""
        assert self.dim == 0
        return float(self.data)

    def __getitem__(self, idx):
        return Functions.Slice(self, idx)

    def backward(self, grad: np.ndarray = None):
        if not self.requires_grad:
            raise NoGradError()

        if grad is None:
            if self.dim > 0:
                raise AutogradError("Grad must be specified for non-zero tensor.")
            grad = np.array(1.0)

        self.grad = self.grad + grad
        for tensor, grad_fn in self.depends_on:
            tensor.backward(grad_fn(grad))

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Functions.Add(self, other)

    def __radd__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Functions.Add(other, self)

    def __iadd__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        if self.requires_grad:
            raise InplaceOpError()
        self.data += other.data

    def __neg__(self):
        return Functions.Neg(self)

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return other + -self

    def __isub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        if self.requires_grad:
            raise InplaceOpError()
        self.data -= other.data

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Functions.Mul(self, other)

    def __rmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Functions.Mul(other, self)

    def __imul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        if self.requires_grad:
            raise InplaceOpError()
        self.data *= other.data

    def __pow__(self, n):
        returned_tensor = 1
        for _ in range(n):
            returned_tensor = returned_tensor * self
        return returned_tensor

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Functions.MatMul(self, other)

    def __rmatmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Functions.MatMul(other, self)

    def sum(self):
        return Functions.TensorSum(self)
