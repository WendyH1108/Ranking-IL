import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from utils import weight_init


class Encoder(nn.Module):
    def __init__(self, obs_dim, dropout=0.0):
        super().__init__()
        self.repr_dim = 64 * 7 * 7

        conv_layers = [
            nn.Conv2d(obs_dim, 32, 8, stride=4),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.Conv2d(64, 64, 3, stride=1),
        ]

        layers = list()
        for conv in conv_layers:
            layers.extend([conv, nn.ReLU()])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        self.convnet = nn.Sequential(*layers)

        self.apply(weight_init)

    def forward(self, obs):
        obs = obs / 255.0
        h = self.convnet(obs)
        # NOTE: for now don't flatten
        return h


class DQNHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h):
        pass


class ForwardDynamics(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h, a):
        pass


class ScaleGrad(torch.autograd.Function):
    """NOTE: From RLPYT: Model component to scale gradients back from layer, without affecting
    the forward pass.  Used e.g. in dueling heads DQN models."""

    @staticmethod
    def forward(ctx, tensor, scale):
        """Stores the ``scale`` input to ``ctx`` for application in
        ``backward()``; simply returns the input ``tensor``."""
        ctx.scale = scale
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Return the ``grad_output`` multiplied by ``ctx.scale``.  Also returns
        a ``None`` as placeholder corresponding to (non-existent) gradient of
        the input ``scale`` of ``forward()``."""
        return grad_output * ctx.scale, None


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1, bias=True):
        super(NoisyLinear, self).__init__()
        self.bias = bias
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.sampling = True
        self.noise_override = None
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features), requires_grad=bias)
        self.bias_sigma = nn.Parameter(torch.empty(out_features), requires_grad=bias)
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        if not self.bias:
            self.bias_mu.fill_(0)
            self.bias_sigma.fill_(0)
        else:
            self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
            self.bias_mu.data.uniform_(-mu_range, mu_range)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        # Self.training alone isn't a good-enough check, since we may need to
        # activate .eval() during sampling even when we want to use noise
        # (due to batchnorm, dropout, or similar).
        # The extra "sampling" flag serves to override this behavior and causes
        # noise to be used even when .eval() has been called.
        if self.noise_override is None:
            use_noise = self.training or self.sampling
        else:
            use_noise = self.noise_override

        if use_noise:
            return F.linear(
                input,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)
