from microtorch.tensor import Tensor, Dependency
from microtorch.utils import Functions, register
import numpy as np


class Function:
    """
    Abstract function class.
    Don't forget to @register your functions!
    """

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


@register()
class Slice(Function):
    def __call__(self, tensor: Tensor, idxs) -> Tensor:
        data = tensor.data[idxs]

        depends_on = None
        if requires_grad := tensor.requires_grad:

            def grad_fn(grad: np.ndarray) -> np.ndarray:
                bigger_grad = np.zeros(tensor.shape)
                bigger_grad[idxs] = grad
                return bigger_grad

            depends_on = [Dependency(tensor, grad_fn)]

        return Tensor(data, requires_grad, depends_on)


@register()
class TensorSum(Function):
    def __call__(self, tensor: Tensor) -> Tensor:
        data = tensor.data.sum()

        requires_grad, depends_on = False, None
        if tensor.requires_grad:

            def grad_fn(grad: np.ndarray) -> np.ndarray:
                return grad * np.ones_like(tensor.data)

            requires_grad = True
            depends_on = [Dependency(tensor, grad_fn)]
        return Tensor(data, requires_grad, depends_on)


@register()
class Add(Function):
    def __call__(self, left: Tensor, right: Tensor) -> Tensor:

        data = left.data + right.data

        depends_on = list()
        if left.requires_grad:

            def left_grad_fn(grad: np.ndarray) -> np.ndarray:
                # Gradient of sum is just one. In this function
                # we simply deal with broadcasting.

                # In case broadcasting simply added dimensions
                for _ in range(grad.ndim - left.dim):
                    grad = grad.sum(axis=0)

                # In case broadcasting used an existing dim that was 1
                # In this case, we keepdims to match grad and tensor shapes
                for didx, dim in enumerate(left.shape):
                    if dim == 1:
                        grad = grad.sum(axis=didx, keepdims=True)

                return grad

            depends_on.append(Dependency(left, left_grad_fn))

        if right.requires_grad:

            def right_grad_fn(grad: np.ndarray) -> np.ndarray:
                # See comments about broadcasting above

                for _ in range(grad.ndim - right.dim):
                    grad = grad.sum(axis=0)

                for didx, dim in enumerate(right.shape):
                    if dim == 1:
                        grad = grad.sum(axis=didx, keepdims=True)

                return grad

            depends_on.append(Dependency(right, right_grad_fn))

        requires_grad = len(depends_on) > 0
        return Tensor(data, requires_grad, depends_on)


@register()
class Neg(Function):
    def __call__(self, tensor: Tensor) -> Tensor:

        data = -tensor.data

        requires_grad, depends_on = False, None
        if tensor.requires_grad:

            def grad_fn(grad: np.ndarray) -> np.ndarray:
                return -grad

            requires_grad = True
            depends_on = [Dependency(tensor, grad_fn)]

        return Tensor(data, requires_grad, depends_on)


@register()
class Mul(Function):
    def __call__(self, left: Tensor, right: Tensor) -> Tensor:

        data = left.data * right.data

        depends_on = list()
        if left.requires_grad:

            def left_grad_fn(grad: np.ndarray) -> np.ndarray:
                grad = grad * right.data

                # See comments about broadcasting in Add
                for _ in range(grad.ndim - left.dim):
                    grad = grad.sum(axis=0)

                for didx, dim in enumerate(left.shape):
                    if dim == 1:
                        grad = grad.sum(axis=didx, keepdims=True)

                return grad

            depends_on.append(Dependency(left, left_grad_fn))

        if right.requires_grad:

            def right_grad_fn(grad: np.ndarray) -> np.ndarray:
                grad = grad * left.data

                # See comments about broadcasting in Add
                for _ in range(grad.ndim - right.dim):
                    grad = grad.sum(axis=0)

                for didx, dim in enumerate(right.shape):
                    if dim == 1:
                        grad = grad.sum(axis=didx, keepdims=True)

                return grad

            depends_on.append(Dependency(right, right_grad_fn))

        requires_grad = len(depends_on) > 0
        return Tensor(data, requires_grad, depends_on)


@register()
class MatMul(Function):
    def __call__(self, left: Tensor, right: Tensor) -> Tensor:
        """
        Implements matrix multiplication left @ right.
        left is   [I1, ..., In, m]
        right is  [m, J1, ..., Jk]
        where Ii and Jj are irrelevant dims
        """

        data = left.data @ right.data

        depends_on = list()
        if left.requires_grad:

            def left_grad_fn(grad: np.ndarray) -> np.ndarray:
                return grad @ right.data.T

            depends_on.append(Dependency(left, left_grad_fn))

        if right.requires_grad:

            def right_grad_fn(grad: np.ndarray) -> np.ndarray:
                return left.data.T @ grad

            depends_on.append(Dependency(right, right_grad_fn))

        requires_grad = len(depends_on) > 0
        return Tensor(data, requires_grad, depends_on)


@register()
class Tanh(Function):
    def __call__(self, tensor: Tensor) -> Tensor:
        data = np.tanh(tensor.data)

        depends_on = None
        if requires_grad := tensor.requires_grad:

            def grad_fn(grad: np.ndarray) -> np.ndarray:
                return grad * (1 - data ** 2)

            depends_on = [Dependency(tensor, grad_fn)]
        return Tensor(data, requires_grad, depends_on)


"""
Functions below are loss functions used in training. 
Maybe export them to a stand-alone class?
"""


@register()
class MSE(Function):
    """Mean Square Error loss function"""

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        if predictions.shape != targets.shape:
            raise ValueError(
                "Predictions and Targets should have the same shape! "
                f"Instead, {predictions.shape=} vs {targets.shape}"
            )
        return ((predictions - targets) ** 2).sum()  # we could optimize this


@register()
class NLLLoss(Function):
    """Negative Log Likelihood loss function"""

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        predictions are [batch_size, n_classes]
        targets is a tensor of target classes
        """
        assert predictions.dim == 2
        batch_size, n_classes = predictions.shape

        assert targets.dim == 1
        assert targets.data.max() < n_classes

        # compute softmax coefficients
        softmaxdata = predictions.data - np.max(predictions.data, axis=1, keepdims=True)
        softmaxdata = np.exp(softmaxdata)
        softmaxdata /= softmaxdata.sum(axis=1, keepdims=True)

        # compute the loss, which is simply -log(p*) where * is the index of target class
        batch_idxs, target_idxs = np.arange(batch_size), targets.data.astype(np.int)
        lossdata = -np.log(softmaxdata[batch_idxs, target_idxs])
        lossdata = lossdata.mean()

        if predictions.requires_grad:

            def grad_fn(grad: np.ndarray) -> np.ndarray:
                # Loss of NLLLoss is very simple once you do the derivations
                grad = grad * softmaxdata.copy()
                grad[batch_idxs, target_idxs] -= 1
                return grad

            return Tensor(lossdata, True, [Dependency(predictions, grad_fn)])

        return Tensor(lossdata)
