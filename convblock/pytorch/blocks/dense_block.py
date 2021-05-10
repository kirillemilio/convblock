""" Contains implementation of Dense Block. """

import math
import numpy as np
import torch

from ..bases import Module
from ..layers import ConvBlock


class DenseBlock(Module):
    """ Dense Block implementation used in DenseNet architecture. """

    def __init__(self,
                 input_shape: 'ArrayLike[int]',
                 layout: str = 'nac',
                 kernel_size: int = 3,
                 bottleneck_factor: float = 4,
                 growth_rate: int = 32,
                 num_layers: int = 12,
                 use_bottleneck: bool = False,
                 block: 'ConvBlock' = None):
        """ Parametrized dense block for DenseNet architecture.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor.
        kernel_size : int, Tuple[int], List[int] or NDArray[int]
            kernel size of convolution opertion. Default is 3.
        bottleneck_factor : int
            number of channels in bottleneck computted as
            growth_rate * bottleneck_factor.
        growth_rate : int
            number of output_channels in convolutional operation.
            Default is 32.
        num_layers : int
            number of convolutions apllied one by one. Bottleneck 1x...x1
            convolutions are not taken into account.
        use_bottleneck : bool
            whether to use bottleneck or not. Default is False.
        block : ConvBlock or None
            basic block class that will be used
            for internal operations description.
            Can be partially applied ConvBlock or None. If None than ConvBlock
            module will be used. Default is None.
        """
        super().__init__(input_shape)
        block = ConvBlock if block is None else block
        self.module_list = torch.nn.ModuleList()
        shape = input_shape
        for i in range(num_layers):
            if use_bottleneck:
                x = block(
                    input_shape=shape, layout=layout * 2,
                    c=dict(filters=(math.ceil(growth_rate * bottleneck_factor),
                                    growth_rate),
                           kernel_size=(1, kernel_size))
                )
            else:
                x = block(
                    input_shape=shape, layout=layout,
                    c=dict(kernel_size=kernel_size, filters=growth_rate)
                )
            self.module_list.append(x)
            in_channels = shape[0]
            shape = np.array(x.output_shape)
            shape[0] += in_channels
        self._output_shape = shape
        self._output_shape[0] = growth_rate
        self.growth_rate = int(growth_rate)
        self.num_layers = int(num_layers)
        self.bottleneck_factor = int(bottleneck_factor)
        self.use_bottleneck = bool(use_bottleneck)

    @property
    def output_shape(self) -> 'NDArray[int]':
        """ Get shape of the output tensor. """
        return np.asarray(self._output_shape, dtype=np.int)

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for dense block module. """
        x = inputs
        for i, module in enumerate(self.module_list):
            if i == (len(self.module_list) - 1):
                return module(x)
            x = torch.cat([x, module(x)], dim=1)
        return x


class DenseBlockShared(Module):
    """ Implementation of DenseBlock with shared connections. """

    def __init__(self,
                 input_shape,
                 layout: str = 'nac',
                 kernel_size: int = 3,
                 bottleneck_factor: int = 4,
                 growth_rate: int = 32,
                 num_layers: int = 12,
                 use_bottleneck: bool = False,
                 block: 'ConvBlock' = None):
        """ Parametrized dense block with shared connections for DenseNet.

        For more detailts see https://arxiv.org/pdf/1707.01629.pdf.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor.
        kernel_size : int, Tuple[int], List[int] or NDArray[int]
            kernel size of convolution opertion. Default is 3.
        bottleneck_factor : int
            number of channels in bottleneck computted as
            growth_rate * bottleneck_factor.
        growth_rate : int
            number of output_channels in convolutional operation.
            Default is 32.
        num_layers : int
            number of convolutions apllied one by one. Bottleneck 1x...x1
            convolutions are not taken into account.
        use_bottleneck : bool
            whether to use bottleneck or not. Default is False.
        block : ConvBlock or None
            basic block class that will be used
            for internal operations description.
            Can be partially applied ConvBlock or None. If None than ConvBlock
            module will be used. Default is None.
        """
        super().__init__(input_shape)
        block = ConvBlock if block is None else block

        self.growth_rate = int(growth_rate)
        self.num_layers = int(num_layers)
        self.bottleneck_factor = int(bottleneck_factor)
        self.use_bottleneck = bool(use_bottleneck)

        self.g_layers = torch.nn.ModuleList()
        self.f_layers = torch.nn.ModuleList()

        shape = input_shape
        for i in range(num_layers):
            if use_bottleneck:
                filters = math.ceil(growth_rate * bottleneck_factor)
            else:
                filters = growth_rate
            x = block(input_shape=shape, layout=layout,
                      c=dict(kernel_size=1, filters=filters))
            y = block(input_shape=x.output_shape, layout=layout,
                      c=dict(kernel_size=kernel_size,
                             filters=growth_rate))
            self.g_layers.append(x)
            self.f_layers.append(y)
            shape = y.output_shape
        self._output_shape = np.array(shape)

    @property
    def output_shape(self) -> 'NDArray[int]':
        """ Get shape of the output tensor. """
        return np.asarray(self._output_shape, dtype=np.int)

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for dense block module. """
        batch_size, _, *sizes = inputs.size()
        if self.use_bottleneck:
            filters = math.ceil(self.growth_rate * self.bottleneck_factor)
        else:
            filters = self.growth_rate

        h = torch.zeros([batch_size, filters, *sizes],
                        device=inputs.device, dtype=inputs.dtype)
        x = inputs
        for i, (g, f) in enumerate(zip(self.g_layers, self.f_layers)):
            h = h + g(x)
            x = f(h)
        return x


class PeleeDenseBlock(Module):

    def __init__(self,
                 input_shape: 'ArrayLike[int]',
                 layout: str = 'cna',
                 kernel_size: int = 3,
                 growth_rate: int = 32,
                 bottleneck_factor: float = 1.0,
                 num_layers: int = 12,
                 block: 'ConvBlock' = None):
        """ Parametrized dense block for DenseNet architecture.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor.
        kernel_size : int, Tuple[int], List[int] or NDArray[int]
            kernel size of convolution opertion. Default is 3.
        growth_rate : int
            number of output_channels in convolutional operation.
            Default is 32.
        num_layers : int
            number of convolutions apllied one by one. Bottleneck 1x...x1
            convolutions are not taken into account.
        bottleneck_factor : float
            number of channels in bottleneck computted as
            growth_rate * bottleneck_factor. Default is 1.0.
        block : ConvBlock or None
            basic block class that will be used
            for internal operations description.
            Can be partially applied ConvBlock or None. If None than ConvBlock
            module will be used. Default is None.
        """
        super().__init__(input_shape)
        block = ConvBlock if block is None else block
        self.modules_x = torch.nn.ModuleList()
        self.modules_y = torch.nn.ModuleList()
        shape = self.input_shape
        for i in range(num_layers):
            filters_x = [math.ceil(growth_rate * bottleneck_factor / 2),
                         math.ceil(growth_rate / 2),
                         math.ceil(growth_rate / 2)]
            filters_y = [math.floor(growth_rate * bottleneck_factor / 2),
                         math.floor(growth_rate / 2)]
            x = block(
                input_shape=shape, layout=layout * 3,
                c=dict(filters=filters_x,
                       kernel_size=(1, kernel_size,
                                    kernel_size))
            )
            y = block(
                input_shape=shape, layout=layout * 2,
                c=dict(filters=filters_y,
                       kernel_size=(1, kernel_size))
            )
            self.modules_x.append(x)
            self.modules_y.append(y)
            in_channels = shape[0] + y.output_shape[0]
            shape = np.array(x.output_shape)
            shape[0] += in_channels
        self._output_shape = shape
        self.growth_rate = int(growth_rate)
        self.num_layers = int(num_layers)
        self.bottleneck_factor = int(bottleneck_factor)
        self.use_bottleneck = bool(use_bottleneck)

    @property
    def output_shape(self) -> 'NDArray[int]':
        """ Get shape of the output tensor. """
        return np.asarray(self._output_shape, dtype=np.int)

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for dense block module. """
        z = inputs
        for i, (module_x, module_y) in enumerate(zip(self.modules_x,
                                                     self.modules_y)):
            z = torch.cat([z, module_x(z), module_y(z)], dim=1)
        return z
