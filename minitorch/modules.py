from minitorch.tensor import Tensor
import numpy as np


class ModuleError(Exception):
    pass


class Parameter(Tensor):
    def __init__(self, *shape):
        super().__init__(np.random.randn(*shape), requires_grad=True)


class Module:
    """A base Module class that tracks parameters and modules, PyTorch style."""

    def __init__(self):
        self._parameters = dict()
        self._modules = dict()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(*args, **kwargs):
        raise NotImplementedError

    def named_parameters(self):
        """Recursively yields name, param from itself and its modules."""
        yield from self._parameters.items()
        for mname, module in self._modules.items():
            for name, param in module.named_parameters():
                yield f"{mname}.{name}", param

    def parameters(self):
        for _, param in self.named_parameters():
            yield param

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            if not hasattr(self, "_parameters"):
                raise ModuleError(
                    "You have to call super().__init__() "
                    "before registering a Parameter in a Module."
                )
            self._parameters[name] = value
        elif isinstance(value, Module):
            if not hasattr(self, "_modules"):
                raise ModuleError(
                    "You have to call super().__init__ "
                    "before registering a Module in a Module."
                )
            self._modules[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):

        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )
