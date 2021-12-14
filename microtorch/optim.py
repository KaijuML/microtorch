import numpy as np


class Optimizer:
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.lr = lr

    def zero_grad(self):
        for param in self.model.parameters():
            param.zero_grad()

    def step(self):
        return NotImplementedError


class SGD(Optimizer):
    def step(self):
        for param in self.model.parameters():
            param.data -= self.lr * param.grad


class Adam(Optimizer):
    def __init__(self, model, lr=0.001, beta1=0.9, beta2=0.99, eps=1e-8):
        super().__init__(model, lr=lr)
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps

        # Intialize moving average and variance for each parameter
        self.ma = {name: 0 for name, _ in self.model.named_parameters()}
        self.mv = {name: 0 for name, _ in self.model.named_parameters()}
        self.t = 0

    def step(self):
        self.t += 1

        for name, param in self.model.named_parameters():
            self.ma[name] = self.beta1 * self.ma[name] + (1 - self.beta1) * param.grad
            self.mv[name] = self.beta2 * self.mv[name] + (1 - self.beta2) * (
                param.grad ** 2
            )

            ma = self.ma[name] / (1 - self.beta1 ** self.t)
            mv = self.ma[name] / (1 - self.beta2 ** self.t)

            param.data -= self.lr * ma / np.sqrt(mv + self.eps)
