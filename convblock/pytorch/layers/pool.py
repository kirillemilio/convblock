""" Contains pytorch pooling modules compatible with ConvBlock interface. """

import math
import numpy as np
import torch
import torch.nn.functional as F

from ..bases import transform_to_int_tuple
from ..bases import ConvModule, Module

from ..utils import compute_direct_output_shape, compute_transposed_output_shape
from ..utils import compute_direct_same_padding, compute_transposed_same_cropping
from ..utils import crop, pad
from ..utils import INT_TYPES, FLOAT_TYPES

from .conv_block import ConvBlock
from .layers import FlattenFunction


class BasePoolLayer(ConvModule):

    def __init__(self, input_shape, kernel_size=3,
                 stride=1, dilation=1, padding='constant'):
        """ Base class for pooling layers generalized for different dims.

        All pooling layers from this module slightly
        extends functionality of original torch.nn.*Pool modules into four
        main aspects:
        1) Shape of the input tensor is passed as argument of constructor.

        2) Shape of the output tensor can be accessed by 'output_shape'
        property of module.

        3) Different padding modes for 'same' mode. It means that there is
        no need to compute padding size for operation to make output tensor
        shape match input tensor's shape.

        4) All arguments of *Pool operations can be lists, tuples or
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
        padding : str or int
            padding mode or padding value.
        """
        super().__init__(input_shape, kernel_size, stride, dilation)
        self.mode = None

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
                                                         self.stride,
                                                         self.dilation)

        _shape = compute_direct_output_shape(self.input_shape[1:],
                                             self.kernel_size,
                                             self.stride,
                                             self.dilation,
                                             self.pad_sizes)

        self._output_shape = np.array([self.input_shape[0],
                                       *_shape], dtype=np.int)

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method.

        Parameters
        ----------
        inputs : Tensor

        Returns
        -------
        Tensor
            result of pooling operation.
        """
        x = pad(inputs, self.pad_sizes,
                mode=self.padding_mode,
                value=self._padding_value)

        pool_args = (x, self.kernel_size,
                     self.stride, 0, self.dilation)
        return self._method(*pool_args)

    def __repr__(self) -> str:
        """ String representation of the module. """
        s = ('{name}(kernel_size={kernel_size}, stride={stride}')
        if tuple(self.pad_sizes) != (0,) * len(self.pad_sizes):
            s += ', padding={padding}'
        if hasattr(self, 'norm_type'):
            s += ', norm_type={norm_type}'
        s += ", mode='{mode}'"
        s += ')'
        values_dict = {
            'kernel_size': self.kernel_size,
            'padding': tuple(self.pad_sizes),
            'stride': self.stride,
            'mode': self.mode,
            'norm_type': self.norm_type if hasattr(self, 'norm_type') else None
        }
        return s.format(name=self.__class__.__name__, **values_dict)


class MaxPool(BasePoolLayer):

    def __init__(self, *args, **kwargs):
        """ MaxPooling module generalized for different dimensions.

        All pooling layers from this module slightly
        extends functionality of original torch.nn.*Pool modules into four
        main aspects:
        1) Shape of the input tensor is passed as argument of constructor.

        2) Shape of the output tensor can be accessed by 'output_shape'
        property of module.

        3) Different padding modes for 'same' mode. It means that there is
        no need to compute padding size for operation to make output tensor
        shape match input tensor's shape. For transposed operations
        'crop' argument can be set to 'True' or 'False'.

        4) All arguments of *Pool operations can be lists, tuples or
        ndarrays of int, np.int, np.int32 or int64 type.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that
            batch dimension is not taken in account.
        kernel_size : int, Tuple[int], List[int] or NDArray[int]
            size of pooling kernel along each dimension.
        stride : int, Tuple[int], List[int] or NDArray[int]
            size of stride along each dimension. Default is 1.
        dilation : int, Tuple[int], List[int] or NDArray[int]
            dilation rate along each dimension. Default is 1.
        padding : str or int
            padding mode. Can be 'constant', 'reflect',
            'replicate', int or None. Default is 'constant'. If has int type
            then value will be used as 'value' argument in 'constant' mode.
        """
        super().__init__(*args, **kwargs)
        self.mode = 'max'

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method.

        Parameters
        ----------
        inputs : pytorch Tensor

        Returns
        -------
        pytorch Tensor
            result of pooling operation.
        """
        x = pad(inputs, self.pad_sizes,
                mode=self.padding_mode,
                value=self._padding_value)

        pool_args = (x, self.kernel_size,
                     self.stride, 0, self.dilation)

        if self.ndims == 2:
            return F.max_pool1d(*pool_args)
        elif self.ndims == 3:
            return F.max_pool2d(*pool_args)
        elif self.ndims == 4:
            return F.max_pool3d(*pool_args)


class AvgPool(BasePoolLayer):

    def __init__(self, *args, **kwargs):
        """ AvgPooling module generalized for different dimensions.

        All pooling layers from this module slightly
        extends functionality of original torch.nn.*Pool modules into four
        main aspects:
        1) Shape of the input tensor is passed as argument of constructor.

        2) Shape of the output tensor can be accessed by 'output_shape'
        property of module.

        3) Different padding modes for 'same' mode. It means that there is
        no need to compute padding size for operation to make output tensor
        shape match input tensor's shape. For transposed operations
        'crop' argument can be set to 'True' or 'False'.

        4) All arguments of *Pool operations can be lists, tuples or
        ndarrays of int, np.int, np.int32 or int64 type.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that
            batch dimension is not taken in account.
        kernel_size : int, Tuple[int], List[int] or NDArray[int]
            size of pooling kernel along each dimension.
        stride : int, Tuple[int], List[int] or NDArray[int]
            size of stride along each dimension. Default is 1.
        dilation : int, Tuple[int], List[int] or NDArray[int]
            dilation rate along each dimension. Default is 1.
        padding : str or int
            padding mode. Can be 'constant', 'reflect',
            'replicate', int or None. Default is 'constant'. If has int type
            then value will be used as 'value' argument in 'constant' mode.
        """
        super().__init__(*args, **kwargs)
        if np.any(self.to_int_array(self.dilation, 'dilation', self.ndims-1) != 1):
            raise NotImplementedError("Argument 'dilation' that is not equal"
                                      + " to 1 is not supported"
                                      + " by AveragePooling layer.")
        self.mode = 'avg'

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method.

        Parameters
        ----------
        inputs : Tensor

        Returns
        -------
        Tensor
            result of pooling operation.
        """
        x = pad(inputs, self.pad_sizes,
                mode=self.padding_mode,
                value=self._padding_value)

        pool_args = (x, self.kernel_size,
                     self.stride, 0)
        if self.ndims == 2:
            return F.avg_pool1d(*pool_args)
        elif self.ndims == 3:
            return F.avg_pool2d(*pool_args)
        elif self.ndims == 4:
            return F.avg_pool3d(*pool_args)


class AdaptiveMaxPool(Module):

    def __init__(self, input_shape, output_size, **kwargs):
        """ Adaptive MaxPooling module generalized for different dimensions.

        All pooling layers from this module slightly
        extends functionality of original torch.nn.*Pool modules into four
        main aspects:
        1) Shape of the input tensor is passed as argument of constructor.

        2) Shape of the output tensor can be accessed by 'output_shape'
        property of module.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that
            batch dimension is not taken in account.
        output_size : Tuple[int], List[int] or NDArray[int]
            output spatial size for adaptive pool layer.
        """
        super().__init__(input_shape)
        self.mode = 'adaptive_max'
        self.output_size = transform_to_int_tuple(output_size,
                                                  'output_size',
                                                  self.ndims - 1)

    @property
    def output_shape(self):
        """ Get output shape of Adaptive MaxPooling module. """
        return np.array([self.input_shape[0], *self.output_size])

    @property
    def stride(self):
        """ Get stride for Adaptive MaxPooling module. """
        return tuple(float(s) for s in (self.input_shape[1:]
                                        / self.output_shape[1:]))

    def __repr__(self) -> str:
        """ String representation of the module. """
        s = '{name}(input_shape={input_shape}, output_shape={output_shape})'
        values_dict = {'input_shape': self.input_shape,
                       'output_shape': self.output_shape}
        return s.format(name=self.__class__.__name__, **values_dict)

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method.

        Parameters
        ----------
        inputs : pytorch Tensor

        Returns
        -------
        pytorch Tensor
            result of pooling operation.
        """
        pool_args = (inputs, self.output_size)

        if self.ndims == 2:
            return F.adaptive_max_pool1d(*pool_args)
        elif self.ndims == 3:
            return F.adaptive_max_pool2d(*pool_args)
        elif self.ndims == 4:
            return F.adaptive_max_pool3d(*pool_args)


class AdaptiveAvgPool(Module):

    def __init__(self, input_shape, output_size, **kwargs):
        """ Adaptive AvgPooling module generalized for different dimensions.

        All pooling layers from this module slightly
        extends functionality of original torch.nn.*Pool modules into four
        main aspects:
        1) Shape of the input tensor is passed as argument of constructor.

        2) Shape of the output tensor can be accessed by 'output_shape'
        property of module.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that
            batch dimension is not taken in account.
        output_size : Tuple[int], List[int] or NDArray[int]
            output spatial size for adaptive pool layer.
        """
        super().__init__(input_shape)
        self.mode = 'adaptive_avg'
        self.output_size = transform_to_int_tuple(output_size,
                                                  'output_size',
                                                  self.ndims - 1)

    @property
    def output_shape(self):
        """ Get output shape of Adaptive MaxPooling module. """
        return np.array([self.input_shape[0], *self.output_size])

    @property
    def stride(self):
        """ Get stride for Adaptive MaxPooling module. """
        return tuple(float(s) for s in (self.input_shape[1:]
                                        / self.output_shape[1:]))

    def __repr__(self) -> str:
        """ String representation of the module. """
        s = '{name}(input_shape={input_shape}, output_shape={output_shape})'
        values_dict = {'input_shape': self.input_shape,
                       'output_shape': self.output_shape}
        return s.format(name=self.__class__.__name__, **values_dict)

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method.

        Parameters
        ----------
        inputs : pytorch Tensor

        Returns
        -------
        pytorch Tensor
            result of pooling operation.
        """
        pool_args = (inputs, self.output_size)

        if self.ndims == 2:
            return F.adaptive_avg_pool1d(*pool_args)
        elif self.ndims == 3:
            return F.adaptive_avg_pool2d(*pool_args)
        elif self.ndims == 4:
            return F.adaptive_avg_pool3d(*pool_args)


class LPPool(BasePoolLayer):

    def __init__(self, input_shape, kernel_size=3, stride=1,
                 padding='constant', norm_type=1.0):
        """ LP-Pooling layer generalized for different dimensions.

        All pooling layers from this module slightly
        extends functionality of original torch.nn.*Pool modules into four
        main aspects:
        1) Shape of the input tensor is passed as argument of constructor.

        2) Shape of the output tensor can be accessed by 'output_shape'
        property of module.

        3) Different padding modes for 'same' mode. It means that there is
        no need to compute padding size for operation to make output tensor
        shape match input tensor's shape. For transposed operations
        'crop' argument can be set to 'True' or 'False'.

        4) All arguments of *Pool operations can be lists, tuples or
        ndarrays of int, np.int, np.int32 or int64 type.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that
            batch dimension is not taken in account.
        kernel_size : int, Tuple[int], List[int] or NDArray[int]
            size of pooling kernel along each dimension.
        stride : int, Tuple[int], List[int] or NDArray[int]
            size of stride along each dimension. Default is 1.
        dilation : int, Tuple[int], List[int] or NDArray[int]
            dilation rate along each dimension. Default is 1.
        padding : str or int
            padding mode. Can be 'constant', 'reflect',
            'replicate', int or None. Default is 'constant'. If has int type
            then value will be used as 'value' argument in 'constant' mode.
        norm_type: float
            defines exponent value of LP-Pooling.
        """
        super().__init__(input_shape, kernel_size, stride, 1, padding)
        if np.any(self.to_int_array(self.dilation, 'dilation', self.ndims-1) != 1):
            raise NotImplementedError("Argument 'dilation' that is not equal"
                                      + " to 1 is not supported"
                                      + " by LP-Pooling layer.")

        if self.ndims == 4:
            raise NotImplementedError("LP-Pooling is not supported for volumetric"
                                      + " input tensors.")
        self.norm_type = float(norm_type)
        self.mode = 'lp'

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for generalize LPPooling layer.

        Parameters
        ----------
        inputs : Tensor
            input tensor.

        Returns
        -------
        Tensor
            result of pooling operation.
        """
        x = pad(inputs, self.pad_sizes,
                mode=self.padding_mode,
                value=self._padding_value)

        if self.ndims == 2:
            pool_args = (x, self.norm_type,
                         self.kernel_size[0], self.stride[0])
            return F.lp_pool1d(*pool_args)
        elif self.ndims == 3:
            pool_args = (x, self.norm_type, self.kernel_size, self.stride)
            return F.lp_pool2d(*pool_args)
        elif self.ndims == 4:
            return F.lp_pool3d(*pool_args)


@ConvBlock.register_option(name='p', vectorized_params=('kernel_size',
                                                        'stride', 'dilation'))
def Pool(input_shape, kernel_size=3, stride=2, dilation=1,
         mode='max', padding='constant', norm_type=1.0, output_size=None):
    """ Generalized pooling layer combines MaxPool, AvgPool and LPPool.

    All pooling layers from this module slightly
    extends functionality of original torch.nn.*Pool modules into four
    main aspects:
    1) Shape of the input tensor is passed as argument of constructor.

    2) Shape of the output tensor can be accessed by 'output_shape'
    property of module.

    3) Different padding modes for 'same' mode. It means that there is
    no need to compute padding size for operation to make output tensor
    shape match input tensor's shape. For transposed operations
    'crop' argument can be set to 'True' or 'False'.

    4) All arguments of *Pool operations can be lists, tuples or
    ndarrays of int, np.int, np.int32 or int64 type.

    Parameters
    ----------
    input_shape : Tuple[int], List[int] or NDArray[int]
        shape of the input tensor. Note that
        batch dimension is not taken in account.
    kernel_size : int, Tuple[int], List[int] or NDArray[int]
        size of pooling kernel along each dimension.
    stride : int, Tuple[int], List[int] or NDArray[int]
        size of stride along each dimension. Default is 1.
    dilation : int, Tuple[int], List[int] or NDArray[int]
        dilation rate along each dimension. Default is 1.
    padding : str or int
        padding mode. Can be 'constant', 'reflect',
        'replicate', int or None. Default is 'constant'. If has int type
        then value will be used as 'value' argument in 'constant' mode.
    norm_type: float
        defines exponent value of LP-Pooling.
    output_size : Tuple[int], List[int] or NDArray[int]
        output spatial size for adaptive pool layer. If not None then
        adaptive mode for pooling will be used and all other
        parameters of layer will be ignored. Default is None
        meaning that adaptive mode is off.
    """
    if not (isinstance(mode, str) and mode in ('max', 'avg', 'lp')):
        raise ValueError("Argument 'mode' must have type str "
                         + "and be one of 'max', 'avg' or 'lp' values.")
    is_adaptive = output_size is not None
    if mode == 'max' and not is_adaptive:
        return MaxPool(input_shape, kernel_size,
                       stride, dilation, padding)
    elif mode == 'max' and is_adaptive:
        return AdaptiveMaxPool(input_shape, output_size)
    elif mode == 'avg' and not is_adaptive:
        return AvgPool(input_shape, kernel_size,
                       stride, dilation, padding)
    elif mode == 'avg' and is_adaptive:
        return AdaptiveAvgPool(input_shape, output_size)
    elif mode == 'lp' and not is_adaptive:
        return LPPool(input_shape, kernel_size, stride, padding, norm_type)
    elif mode == 'lp':
        raise ValueError("Adaptive mode is not available for lp-pooling.")


@ConvBlock.register_option('g')
@ConvBlock.register_option('>')
class GlobalPool(Module):

    def __init__(self, input_shape, mode='avg'):
        """ Global pooling layer generalized for different dimensions.

        Parameters
        ----------
        input_shape : tuple, list or ndarray of ints
            shape of the input tensor (batch dimension is not taken into account).
        mode : str
            mode of pooling. Can be 'max' or 'avg'. Default is 'avg'.
        """
        super().__init__(input_shape)
        if mode not in ('max', 'avg'):
            raise ValueError("Argument 'mode' must be 'max' or 'avg'")
        self.mode = mode
        self.layer = Pool(input_shape, kernel_size=input_shape[1:],
                          stride=input_shape[1:], padding=None, mode=mode)

    def forward(self, inputs):
        """ Forward pass method for global pooling layer.

        Parameters
        ----------
        inputs : Tensor

        Returns
        -------
        Tensor
            result of pooling operation.
        """
        x = self.layer(inputs)
        return FlattenFunction.apply(x)

    @property
    def output_shape(self) -> 'NDArray[int]':
        """ Get shape of the output tensor. """
        return np.array([self.input_shape[0]])

    def __repr__(self) -> str:
        """ String representation of Global Pooling module. """
        s = "{name}(input_shape={input_shape}, "
        s += "output_shape={output_shape}, "
        s += "mode='{mode}')"
        return s.format(name='GlobalPool', mode=self.mode,
                        input_shape=tuple(self.input_shape),
                        output_shape=self.output_shape[0])
