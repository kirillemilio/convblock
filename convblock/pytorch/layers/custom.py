""" Contains implementation of PixelScaleFunction and PixelScaler classes. """

import numpy as np
import torch

from ..utils import INT_TYPES, FLOAT_TYPES
from ..utils import transform_to_float_tuple
from ..bases import Module


class ChannelsShuffleFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, permutation):
        input = input.detach()
        ctx.permutation = permutation
        return input[:, permutation, ...]

    @staticmethod
    def backward(ctx, grad_output):
        _, permutation = ctx.permutation.sort()
        return grad_output[:, permutation], None

    @staticmethod
    def symbolic(g, input_tensor, permutation):
        r = g.op("ChannelsShuffleFunction", input_tensor, permutation)
        return r


class PixelScaleFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, scale, bias, compute_grad=False):
        num_dims = len(input.shape)
        if num_dims == 5:
            permutation = (0, 2, 3, 4, 1)
            inv_permutation = (0, 4, 1, 2, 3)
        if num_dims == 4:
            permutation = (0, 2, 3, 1)
            inv_permutation = (0, 3, 1, 2)
        elif num_dims == 3:
            permutation = (0, 2, 1)
            inv_permutation = (0, 2, 1)
        input = input.detach()
        ctx.input = input
        ctx.scale = scale.detach()
        ctx.bias = bias.detach()
        ctx.compute_grad = compute_grad
        x = input.detach().permute(*permutation)
        y = x * scale + bias
        return y.permute(*inv_permutation)

    @staticmethod
    def backward(ctx, grad_output):
        num_dims = len(grad_output.shape)
        if num_dims == 5:
            permutation = (0, 2, 3, 4, 1)
            inv_permutation = (0, 4, 1, 2, 3)
        if num_dims == 4:
            permutation = (0, 2, 3, 1)
            inv_permutation = (0, 3, 1, 2)
        elif num_dims == 3:
            permutation = (0, 2, 1)
            inv_permutation = (0, 2, 1)
        y = grad_output.detach().permute(*permutation)
        y = y * ctx.scale
        input_grad = y.permute(*inv_permutation)

        if ctx.compute_grad:
            scale_grad = (
                (grad_output * ctx.input)
                .permute(*permutation)
                .contiguous()
                .view(-1, ctx.input.shape[1])
                .sum(0)
            )
            bias_grad = (
                grad_output
                .permute(*permutation)
                .contiguous()
                .view(-1, ctx.input.shape[1])
                .sum(0)
            )

            return input_grad, scale_grad, bias_grad, None
        else:
            return input_grad, None, None, None

    @staticmethod
    def symbolic(g, input_tensor, scale, bias, compute_grad=True):
        r = g.op("PixelScale", input_tensor, scale, bias)
        return r


class PixelScaler(Module):

    def __init__(self, input_shape, scale=1, bias=0, compute_grad=False):
        """ Pixel scaling layer.

        Parameters
        ----------
        input_shape : ArrayLike[int]
            shape of the input tensor. Note that batch dimension is not
            taken into account.
        scale : float or ArrayLike[float]
            scale for each channel of input tensor.
        bias : float or ArrayLike[float]
            bias for each channel of input tensor.
        compute_grad : bool
            whether to compute scale and bias parameters gradients.
            Default is False.
        """
        super().__init__(input_shape=input_shape)

        if isinstance(scale, INT_TYPES + FLOAT_TYPES):
            scale = np.array([scale] * self.in_channels)
        else:
            scale = np.array(scale, dtype='float32')

        if isinstance(bias, INT_TYPES + FLOAT_TYPES):
            bias = np.array([bias] * self.in_channels)
        else:
            bias = np.array(bias, dtype='float32')

        bias = torch.from_numpy(bias).type(torch.FloatTensor)
        scale = torch.from_numpy(scale).type(torch.FloatTensor)

        if compute_grad:
            self.bias = torch.nn.Parameter(bias)
            self.scale = torch.nn.Parameter(scale)
            self.compute_grad = True

        else:
            self.register_buffer('scale', scale)
            self.register_buffer('bias', bias)
            self.compute_grad = False

    def forward(self, input_tensor: 'Tensor') -> 'Tensor':
        """ Forward pass method for pixel scaling. """
        return PixelScaleFunction.apply(input_tensor, self.scale, self.bias,
                                        self.compute_grad)

    def __repr__(self) -> str:
        """ String representation of upsampling layer. """
        s = "{name}(input_shape={input_shape}, "
        s += "scale={scale}, "
        s += "bias={bias}"
        if self.compute_grad:
            s += ", compute_grad=True"
        s += ")"
        return s.format(name=self.__class__.__name__,
                        input_shape=tuple(self.input_shape),
                        scale=tuple(self.scale.data.cpu().numpy()),
                        bias=tuple(self.bias.data.cpu().numpy()),
                        compute_grad=self.compute_grad)
