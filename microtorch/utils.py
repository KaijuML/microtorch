class AutogradError(Exception):
    pass


class NoGradError(AutogradError):
    def __init__(self):
        msg = "Cannot access grad of a Tensor with requires_grad=False!"
        super().__init__(msg)


class InplaceOpError(AutogradError):
    def __init__(self):
        msg = "No inplace operations when tracking gradients!"
        super().__init__(msg)


class Functions:
    """
    To be used later by autograd.Tensor & Cie.
    In practice, create a new function with the @register decorator
    to add your function to this master class (see examples in functions.py).
    """

    pass


def register():
    def _register(fn):
        setattr(Functions, fn.__name__, fn())
        return fn

    return _register
