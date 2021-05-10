""" Contains pytorch convolutional layers compatible with ConvBlock interface. """

import math
import numpy as np
import torch
import torch.nn.functional as F

from ..bases import transform_to_int_tuple
from ..bases import ConvModule

from ..utils import compute_direct_output_shape
from ..utils import compute_transposed_output_shape
from ..utils import compute_direct_same_padding
from ..utils import compute_transposed_same_cropping
from ..utils import crop, pad
from ..utils import INT_TYPES, FLOAT_TYPES

from .conv_block import ConvBlock


def map_initializer(x):
    if x == 'zeros':
        return torch.nn.init.zeros_
    elif x == 'ones':
        return torch.nn.init.ones_
    elif x == 'constant':
        return torch.nn.init.constant_
    elif x in ['xnormal', 'xn',
               'xavier_normal',
               'glorot_normal']:
        return torch.nn.init.xavier_normal_
    elif x in ['xuniform', 'xu',
               'xavier_uniform',
               'glorot_uniform']:
        return torch.nn.init.xavier_uniform_
    elif x == 'uniform' or x == 'u':
        return torch.nn.init.uniform_
    elif x == 'normal' or x == 'n':
        return torch.nn.init.normal_
    else:
        raise ValueError("Unknown initializer: '{}'".format(x))


class BaseConvLayer(ConvModule):

    def __init__(self, input_shape, filters, kernel_size=3,
                 stride=1, dilation=1, groups=1, bias=False,
                 init_weight=None, init_bias=None):
        """ Base class for convolutional layers generalized for different dims.

        All convolutional layers from this module slightly
        extends functionality of original torch.nn.Conv* modules in four
        main aspects:
        1) Shape of the input tensor is passed as argument of constructor.

        2) Shape of the output tensor can be accessed by 'output_shape'
        property of the Conv* module.

        3) Different padding modes for 'same' mode. It means that there is
        no need to compute padding size for operation to make output tensor
        shape match input tensor's shape.

        4) All arguments of Conv* operations can be lists, tuples or
        ndarrays of int, np.int, np.int32 or int64 type.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that
            batch dimension is not taken in account.
        filters : int
            number of channels in the output tensor.
        kernel_size : int, Tuple[int], List[int] or NDArray[int]
            size of convolution or deconvolution kernel along each dimension.
        stride : int, Tuple[int], List[int] or NDArray[int]
            size of stride along each dimension. Default is 1.
        dilation : int, Tuple[int], List[int] or NDArray[int]
            dilation rate along each dimension. Default is 1.
        groups : int
            number of groups. Default is 1.
        bias : bool
            whether to use bias or not. Default is False.
        init_weight : callable, str, dict or None
            weight variable initializer. There are several options when passing
            custom initializer to this layer
            1) If str then must be one of
               following values: ['x', 'u', 'xu', 'xn', 'xavier_normal',
               'xavier_uniform', 'glorot_normal', 'glorot_uniform', 'zeros',
               'ones', 'constant', 'uniform', 'normal'].
            2) If callable then initialization must be performed inplace.
            3) If dict then must contain at least two key-value pairs:
               'init' is a callable that performs initialization.
               'inplace' whether initialization is performed inplace.
               Default is True meaning that initialization is performed
               inplace.    
        init_bias : callable, str, dict or None
            bias variable initializer. There are several options when passing
            custom initializer to this layer
            1) If str then must be one of
               following values: ['x', 'u', 'xu', 'xn', 'xavier_normal',
               'xavier_uniform', 'glorot_normal', 'glorot_uniform', 'zeros',
               'ones', 'constant', 'uniform', 'normal'].
            2) If callable then initialization must be performed inplace.
            3) If dict then must contain at least two key-value pairs:
               'init' is a callable that performs initialization.
               'inplace' whether initialization is performed inplace.
               Default is True meaning that initialization is performed
               inplace.
        """
        super().__init__(input_shape, kernel_size, stride, dilation)
        self.filters = int(filters)
        self.groups = int(groups)

        if isinstance(init_weight, np.ndarray):
            weight_param = torch.nn.Parameter(torch.from_numpy(init_weight))
            self._weight_init = 'NDArray({})'.format(weight_param.shape)
        elif isinstance(init_weight, torch.nn.Parameter):
            weight_param = init_weight
            self._weight_init = 'Parameter({})'.format(weight_param.shape)
        elif isinstance(init_weight, torch.Tensor):
            weight_param = torch.nn.Parameter(init_weight)
            self._weight_init = 'Tensor({})'.format(weight_param.shape)
        elif callable(init_weight) or init_weight is None:
            init_weight = {'init': init_weight, 'inplace': True}
            self._weight_init = init_weight
        else:
            weight_param = None
            init_weight = dict(init_weight)
            assert init_weight.contains('init')
            assert callable(init_weight.get('init'))
            if 'inplace' not in init_weight:
                init_weight['inplace'] = True
            self._weight_init = init_weight.get('init')

        if not isinstance(init_weight, dict):
            init_weight = {
                'init': weight_param,
                'inplace': True
            }

        if isinstance(init_bias, np.ndarray):
            bias_param = torch.nn.Parameter(torch.from_numpy(init_bias))
            self._bias_init = 'NDArray({})'.format(bias_param.shape)
        elif isinstance(init_bias, torch.nn.Parameter):
            bias_param = init_bias
            self._bias_init = 'Parameter({})'.format(bias_param.shape)
        elif isinstance(init_bias, torch.Tensor):
            bias_param = torch.nn.Parameter(init_bias)
            self._bias_init = 'Tensor({})'.format(bias_param.shape)
        elif callable(init_bias) or init_bias is None:
            init_bias = {'init': init_bias, 'inplace': True}
            self._bias_init = init_bias
        else:
            bias_param = None
            init_bias = dict(init_bias)
            assert init_bias.contains('init')
            assert callable(init_bias.get('init'))
            if 'inplace' not in init_bias:
                init_bias['inplace'] = True
            self._bias_init = init_bias.get('init')

        if not isinstance(init_bias, dict):
            init_bias = {
                'init': bias_param,
                'inplace': True
            }

        self.weight = self._create_weight(**init_weight)
        if bias:
            self.bias = self._create_bias(**init_bias)
        else:
            self.register_parameter('bias', None)

    def _create_bias(self, init=None, inplace=True, **kwargs) -> 'torch.nn.Parameter':
        """ Create bias tensor for conv operation and wrap it as Parameter. """
        if isinstance(init, torch.nn.Parameter):
            assert len(init.size()) == 1
            assert init.size(0) == self.filters
            return init
        bias = torch.nn.Parameter(torch.Tensor(self.filters))
        if init is None:
            n = self.in_channels
            for k in self.kernel_size:
                n *= k
            stdv = 1. / math.sqrt(n)
            bias.data.uniform_(-stdv, stdv)

        elif callable(init) and inplace:
            init(bias, **kwargs)
        elif callable(init) and not inplace:
            bias = init(bias, **kwargs)
        elif isinstance(init, str):
            map_initializer(init)(bias, **kwargs)
        return bias

    def _create_weight(self, init=None, inplace=True, **kwargs) -> 'torch.nn.Parameter':
        """ Create weigth tensor for conv operation and wrap is as Parameter. """
        if isinstance(init, torch.nn.Parameter):
            assert len(init.size()) == len(self.kernel_size) + 2
            assert init.size(0) == self.filters
            assert init.size(1) == self.in_channels // self.groups
            for i, k in enumerate(self.kernel_size):
                assert init.size(2 + i) == k
            return init
        weight = torch.nn.Parameter(
            torch.Tensor(self.filters,
                         self.in_channels // self.groups,
                         *self.kernel_size)
        )
        if init is None:
            n = self.in_channels
            for k in self.kernel_size:
                n *= k
            stdv = 1. / math.sqrt(n)
            weight.data.uniform_(-stdv, stdv)
        elif callable(init) and inplace:
            init(weight, **kwargs)
        elif callable(init) and not inplace:
            weight = init(weight, **kwargs)
        elif isinstance(init, str):
            map_initializer(init)(weight, **kwargs)
        return weight


@ConvBlock.register_option(name='c', vectorized_params=('kernel_size',
                                                        'stride', 'dilation'))
class Conv(BaseConvLayer):

    def __init__(self, input_shape, filters, kernel_size=3, stride=1,
                 dilation=1, groups=1, padding='constant', bias=False,
                 init_weight=None, init_bias=None):
        """ Direct convolution layer generalized for different dims.

        This layer slightly extends functionality of original
        torch.nn.Conv* modules in four
        main aspects:

        1) Shape of the input tensor is passed as argument of constructor.

        2) Shape of the output tensor can be accessed by 'output_shape'
        property of the Conv* module.

        3) Different padding modes for 'same' mode. It means that there is
        no need to compute padding size for operation to make output tensor
        shape match input tensor's shape.

        4) All arguments of Conv* operations can be lists, tuples or
        ndarrays of int, np.int, np.int32 or int64 type.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that
            batch dimension is not taken in account.
        filters : int
            number of channels in the output tensor.
        kernel_size : int, Tuple[int], List[int] or NDArray[int]
            size of convolution kernel along each dimension.
        stride : int, Tuple[int], List[int] or NDArray[int]
            size of stride along each dimension. Default is 1.
        dilation : int, Tuple[int], List[int] or NDArray[int]
            dilation rate along each dimension. Default is 1.
        groups : int
            number of groups. Default is 1.
        padding : str or int
            padding mode. Can be 'constant', 'reflect',
            'replicate', int or None. Default is 'constant'. If has int type
            then value will be used as 'value' argument in 'constant' mode.
        bias : bool
            whether to use bias or not. Default is False.
        init_weight : callable, str, dict or None
            weight variable initializer. There are several options
            when passing custom initializer to this layer:
            1) If str then must be one of
               following values: ['x', 'u', 'xu', 'xn', 'xavier_normal',
               'xavier_uniform', 'glorot_normal', 'glorot_uniform', 'zeros',
               'ones', 'constant', 'uniform', 'normal'].
            2) If callable then initialization must be performed inplace.
            3) If dict then must contain at least two key-value pairs:
               'init' is a callable that performs initialization.
               'inplace' whether initialization is performed inplace.
               Default is True meaning that initialization is performed
               inplace.    
        init_bias : callable, str, dict or None
            bias variable initializer. There are several options
            when passing custom initializer to this layer:
            1) If str then must be one of
               following values: ['x', 'u', 'xu', 'xn', 'xavier_normal',
               'xavier_uniform', 'glorot_normal', 'glorot_uniform', 'zeros',
               'ones', 'constant', 'uniform', 'normal'].
            2) If callable then initialization must be performed inplace.
            3) If dict then must contain at least two key-value pairs:
               'init' is a callable that performs initialization.
               'inplace' whether initialization is performed inplace.
               Default is True meaning that initialization is performed
               inplace.
        """
        super().__init__(input_shape, filters, kernel_size,
                         stride, dilation, groups, bias,
                         init_weight, init_bias)

        if isinstance(padding, (*INT_TYPES, *FLOAT_TYPES)):
            self.padding_mode = 'constant'
            self._padding_value = padding
        else:
            self.padding_mode = padding
            self._padding_value = 0.0

        if self.padding_mode == 'valid' or self.padding_mode is None:
            self.pad_sizes = [0] * (self.ndims - 1) * 2
        else:
            self.pad_sizes = compute_direct_same_padding(self.kernel_size,
                                                         self._stride,
                                                         self.dilation)

        _shape = compute_direct_output_shape(self.input_shape[1:],
                                             self.kernel_size,
                                             self._stride,
                                             self.dilation,
                                             self.pad_sizes)

        self._output_shape = np.array([self.filters, *_shape], dtype=np.int)

    def __repr__(self) -> str:
        """ String representation of convolutional layer. """
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if tuple(self.pad_sizes) != (0,) * len(self.pad_sizes):
            s += ', padding={padding}'
            s += ', padding_mode={padding_mode}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self._bias_init is not None and self._bias_init.get('init') is not None:
            s += ', init_bias={init_bias}'
        if self._weight_init is not None and self._weight_init.get('init') is not None:
            s += ', init_weight={init_weight}'
        s += ')'
        values_dict = {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'padding': tuple(self.pad_sizes),
            'padding_mode': self.padding_mode,
            'groups': self.groups,
            'bias': self.bias,
            'dilation': self.dilation,
            'stride': self._stride,
            'init_bias': self._bias_init,
            'init_weight': self._weight_init
        }
        return s.format(name=self.__class__.__name__, **values_dict)

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for tranposed convolution layer.

        Parameters
        ----------
        inputs : Tensor
            input tensor for transposed convolution layer.

        Returns
        -------
        Tensor
            result of convolutional operation applied to the input tensor.
        """
        x = pad(inputs, self.pad_sizes,
                mode=self.padding_mode,
                value=self._padding_value)
        conv_args = (x, self.weight, self.bias, self._stride, 0,
                     self.dilation, self.groups)

        if self.ndims == 2:
            return F.conv1d(*conv_args)
        elif self.ndims == 3:
            return F.conv2d(*conv_args)
        elif self.ndims == 4:
            return F.conv3d(*conv_args)


@ConvBlock.register_option(name='t', vectorized_params=('kernel_size',
                                                        'stride',
                                                        'dilation'))
class ConvTransposed(BaseConvLayer):

    def _create_weight(self, init=None, inplace=True, **kwargs) -> 'torch.nn.Parameter':
        """ Create weigth tensor for conv operation and wrap is as Parameter. """
        weight = torch.nn.Parameter(
            torch.Tensor(self.in_channels // self.groups,
                         self.filters, *self.kernel_size)
        )
        if init is None:
            n = self.in_channels
            for k in self.kernel_size:
                n *= k
            stdv = 1. / math.sqrt(n)
            weight.data.uniform_(-stdv, stdv)
        elif callable(init) and inplace:
            init(weight, **kwargs)
        elif callable(init) and not inplace:
            weight = init(weight, **kwargs)
        elif isinstance(init, str):
            map_initializer(init)(weight, **kwargs)
        return weight

    @property
    def stride(self):
        return tuple(float(1.0 / s) for s in self._stride)

    def __init__(self, input_shape, filters, kernel_size=3,
                 stride=1, dilation=1, groups=1, crop=True,
                 bias=False, init_weight=None, init_bias=None):
        """ Transposed convolution layer generalized for different dims.

        This layer slightly extends functionality of original
        torch.nn.TransposedConv* modules in four
        main aspects:

        1) Shape of the input tensor is passed as argument of constructor.

        2) Shape of the output tensor can be accessed by 'output_shape'
        property of the Conv* module.

        3) Different padding modes for 'same' mode. It means that there is
        no need to compute padding size for operation to make output tensor
        shape match input tensor's shape.

        4) All arguments of Conv* operations can be lists, tuples or
        ndarrays of int, np.int, np.int32 or int64 type.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that
            batch dimension is not taken in account.
        filters : int
            number of channels in the output tensor.
        kernel_size : int, Tuple[int], List[int] or NDArray[int]
            size of deconvolution kernel along each dimension.
        stride : int, Tuple[int], List[int] or NDArray[int]
            size of stride along each dimension. Default is 1.
        dilation : int, Tuple[int], List[int] or NDArray[int]
            dilation rate along each dimension. Default is 1.
        groups : int
            number of groups. Default is 1.
        crop : bool
            whether to crop output tensor to have the same spatial
            shape as input tensor or not. Default is True.
        bias : bool
            whether to use bias or not. Default is False.
        init_weight : callable, str, dict or None
            weight variable initializer. There are several options
            when passing custom initializer to this layer:
            1) If str then must be one of
               following values: ['x', 'u', 'xu', 'xn', 'xavier_normal',
               'xavier_uniform', 'glorot_normal', 'glorot_uniform', 'zeros',
               'ones', 'constant', 'uniform', 'normal'].
            2) If callable then initialization must be performed inplace.
            3) If dict then must contain at least two key-value pairs:
               'init' is a callable that performs initialization.
               'inplace' whether initialization is performed inplace.
               Default is True meaning that initialization is performed
               inplace.    
        init_bias : callable, str, dict or None
            bias variable initializer. There are several options when passing
            custom initializer to this layer:
            1) If str then must be one of
               following values: ['x', 'u', 'xu', 'xn', 'xavier_normal',
               'xavier_uniform', 'glorot_normal', 'glorot_uniform', 'zeros',
               'ones', 'constant', 'uniform', 'normal'].
            2) If callable then initialization must be performed inplace.
            3) If dict then must contain at least two key-value pairs:
               'init' is a callable that performs initialization.
               'inplace' whether initialization is performed inplace.
               Default is True meaning that initialization is performed
               inplace.
        """
        super().__init__(input_shape, filters, kernel_size,
                         stride, dilation, groups, bias,
                         init_weight, init_bias)
        self.crop = crop
        if self.crop:
            self.crop_sizes = compute_transposed_same_cropping(self.kernel_size,
                                                               self._stride,
                                                               self.dilation)
        else:
            self.crop_sizes = [0] * (self.ndims - 1) * 2

        _shape = compute_transposed_output_shape(self.input_shape[1:],
                                                 self.kernel_size,
                                                 self._stride,
                                                 self.dilation,
                                                 self.crop_sizes)

        self._output_shape = np.array([self.filters, *_shape], dtype=np.int)

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for transposed convolution layer.

        Parameters
        ----------
        inputs : torch Tensor
            input tensor for transposed convolution layer.

        Returns
        -------
        torch Tensor
            result of convolutional operation applied to the input tensor.
        """
        conv_args = (inputs, self.weight, self.bias,
                     self._stride, 0, 0, self.groups, self.dilation)

        if self.ndims == 2:
            x = F.conv_transpose1d(*conv_args)
        elif self.ndims == 3:
            x = F.conv_transpose2d(*conv_args)
        elif self.ndims == 4:
            x = F.conv_transpose3d(*conv_args)

        if self.crop:
            x = crop(x, self.crop_sizes)
        return x

    def __repr__(self) -> 'str':
        """ String representation of transposed convolution layer. """
        s = ('{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if tuple(self.crop_sizes) != (0,) * len(self.crop_sizes):
            s += ', cropping={cropping}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self._bias_init is not None and self._bias_init.get('init') is not None:
            s += ', init_bias={init_bias}'
        if self._weight_init is not None and self._weight_init.get('init') is not None:
            s += ', init_weight={init_weight}'
        s += ')'
        values_dict = {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'cropping': tuple(self.crop_sizes),
            'groups': self.groups,
            'bias': self.bias,
            'dilation': self.dilation,
            'stride': self._stride,
            'init_bias': self._bias_init,
            'init_weight': self._weight_init
        }
        return s.format(name=self.__class__.__name__, **values_dict)
