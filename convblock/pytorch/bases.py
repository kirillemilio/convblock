""" Contains implementation of MetaModule, Module, Sequential and Layer classes."""

from collections import defaultdict
import numpy as np
import torch

from .utils import transform_to_int_tuple
from .utils import INT_TYPES
from .utils import FLOAT_TYPES
from .utils import LIST_TYPES
from .utils import crop_as
from .utils import pad_as
from .utils import merge


class MetaModule(type):

    """ Metaclass for partial constructor calling. """

    _counter = defaultdict(int)

    def partial(cls, *args, **kwargs):
        """ Build partialy applied module. """

        if hasattr(cls, '_is_partial_module'):
            _kwargs = {**cls._partial_kwargs, **kwargs}
            _args = cls._partial_args if len(args) == 0 else args
            cls = cls._base_cls
        else:
            _kwargs = kwargs
            _args = args

        cls._counter[cls.__name__] += 1

        class PartialModule(cls):
            _is_partial_module = True
            _partial_args = _args
            _partial_kwargs = _kwargs
            _options = cls._options
            _options_params = cls._options_params
            _base_cls = cls
        PartialModule.__name__ = cls.__name__ + \
            '_' + str(cls._counter[cls.__name__])
        return PartialModule

    def p(cls, *args, **kwargs):
        return cls.partial(*args, **kwargs)

    def __mul__(cls, other):
        if not isinstance(other, int):
            raise TypeError
        return MulModule.p(module=cls,
                           num_repeats=other)

    def __or__(cls, other):
        if not isinstance(other, MetaModule):
            raise TypeError
        return OrModule.p(modules=[cls, other])

    def __ror__(cls, other):
        return cls.__or__(other)

    def __rmul__(cls, other):
        return cls.__mul__(other)
    
    def __rrshift__(cls, other):
        if isinstance(other, (Module, Sequential)):
            return Sequential(other, cls(input_shape=other.output_shape))
        else:
            print(type(other))
        assert isinstance(other, (list, tuple))
        other = list(map(int, other))
        return cls(input_shape=other)

    def __new__(mtcls, name, bases, attrs):  # noqa: N804
        attrs = {'_options': {},
                 '_instance_counter': 0,
                 '_options_params': {}, **attrs}
        cls = super(MetaModule, mtcls).__new__(mtcls, name, bases, attrs)
        return cls

    def __call__(cls, *args, **kwargs):
        if hasattr(cls, '_is_partial_module'):
            args = cls._partial_args

            prev_keys = set(cls._partial_kwargs.keys())
            keys = set(kwargs.keys())

            new_kwargs = {}
            for key in keys & prev_keys:
                if (isinstance(cls._partial_kwargs[key], dict)
                        and isinstance(kwargs[key], dict)):

                    new_kwargs[key] = {**cls._partial_kwargs[key],
                                       **kwargs[key]}
                else:
                    new_kwargs[key] = kwargs[key]

            kwargs = {**cls._partial_kwargs, **kwargs, **new_kwargs}
            return cls._base_cls(*args, **kwargs)

        instance = super().__call__(*args, **kwargs)
        cls._instance_counter += 1
        return instance


class Module(torch.nn.Module, metaclass=MetaModule):

    def __init__(self, input_shape):
        super().__init__()
        try:
            input_shape = transform_to_int_tuple(input_shape,
                                                 'input_shape',
                                                 len(input_shape))
        except ValueError:
            input_shape = np.array(input_shape, dtype=int)
            if input_shape.ndim > 2:
                raise ValueError("Argument 'input_shape'"
                                 + " must be 1d or 2d array")
        self._input_shape = np.array(input_shape)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def stride(self) -> 'Tuple[float]':
        """ Get stride associated with layer.

        Returns
        -------
        Tuple[float] or None
            tuple of size ndims - 1 or None if stride
            does not exist for current layer.
        """
        if self.input_shape.ndim > 1:
            return None
        elif len(self.input_shape) - len(self.output_shape) != 0:
            return None
        return (1, ) * (self.ndims - 1)

    @property
    def num_inputs(self) -> int:
        """ Get number of input tensors for module.

        Returns
        -------
        int
            number of output tensors.
        """
        if self.input_shape.ndim == 1:
            return 1
        return self.input_shape.shape[0]

    @property
    def num_outputs(self) -> int:
        """ Get number of output tensors for module.

        Returns
        -------
        int
            number of output tensors.
        """
        if self.output_shape.ndim == 1:
            return 1
        return self.output_shape.shape[0]

    @property
    def ndims(self) -> int:
        """ Number of dimensions in input tensor.

        Batch dimension is not taken into account.
        """
        if self.input_shape.ndim == 2:
            return int(self._input_shape.shape[1])
        return len(self._input_shape)

    @property
    def input_shape(self) -> 'ndarray(int)':
        """ Get shape of the input tensor.

        Returns
        -------
        ndarray(int)
            shape of the input tensor (1d array)
            or shapes of input tensors (2d array).
        """
        return np.array(self._input_shape, dtype=np.int)

    @property
    def output_shape(self):
        """ Get shape of the output tensor.

        Returns
        -------
        ndarray(int)
            shape of the output tensor.
        """
        return self.input_shape

    @property
    def in_channels(self):
        """ Get number of channels in the input tensor of operation.

        Returns
        -------
        int or ndarray(int)
            size of channels dimension of the input tensor.
        """
        if self.input_shape.ndim == 2:
            return self.input_shape[:, 0]
        return int(self.input_shape[0])

    @property
    def out_channels(self):
        """ Get number of channels in the output tensor of operation.

        Returns
        -------
        int or ndarray(int)
            size of channels dimension in the output tensor.
        """
        if self.output_shape.ndim == 2:
            return self.output_shape[:, 0]
        return int(self.output_shape[0])

    def to_int_array(self, parameter, name, length):
        """ Transform input parameter value to int list of given length.

        Parameters
        ----------
        parameter : int, tuple(int), list(int) or ndarray(int)
            input parameter value.
        name : str
            name of parameter. Required by exception raising part of function.
        length : int
            length of output list with parameter values.

        Returns
        -------
        ndarray(int)

        Raises
        ------
        ValueError
            If input parameter has wrong type or has improper length(if list-like).
        """
        if isinstance(parameter, INT_TYPES):
            parameter = np.asarray([parameter] * length, dtype=np.int)
        elif isinstance(parameter, LIST_TYPES):
            parameter = np.asarray(parameter, dtype=np.int).flatten()
            if len(parameter) != length:
                raise ValueError("Argument {} has inproper lenght.".format(name)
                                 + " Must have {}, got {}.".format(length,
                                                                   len(parameter)))
        else:
            raise ValueError("Argument {} must be int or ".format(name)
                             + "tuple, list, ndarray "
                             + "containing {} int values.".format(length))
        return parameter

    @classmethod
    def merge(cls, x, y, how='+'):
        """ Merge tensors according given rule.

        Parameters
        ----------
        x : Tensor
            first tensor.
        y : Tensor
            second tensor.
        how : str
            how to merge input tensors. Can be on of following values:
            '+' for sum, '*' for product or '.' for concatenation along first
            dimension. Default is '+'.

        Returns
        -------
        Tensor
            result of merging operation.

        Raises
        ------
        ValueError
            if argument 'how' has value diverging from '+', '*' or '.'.
        """
        if how not in ('+', '*', '.'):
            raise ValueError("Argument 'how' must be one of "
                             + "following values: ('+', '.', '*'). "
                             + "Got {}.".format(how))
        return merge([x, y], how)

    @classmethod
    def crop_as(cls, x: 'Tensor', y: 'Tensor') -> 'Tensor':
        """ Crop first tensor to have the same shape as the second.

        This method affects only spatial dimensions of tensor, so
        batch size and channels dimension remain unchanged.

        Parameters
        ----------
        x : Tensor
            tensor to crop.
        y : Tensor
            tensor whose shape will be used for cropping.

        Returns
        -------
        Tensor
        """
        return crop_as(x, y)

    @classmethod
    def pad_as(cls, x: 'Tensor', y: 'Tensor') -> 'Tensor':
        """ Add padding to first tensor to have the same shape as the second.

        This method affects only spatial dimensions of tensor, so
        batch size and channels dimension remain unchanged.


        Parameters
        ----------
        x : Tensor
            tensor to pad.
        y : Tensor
            tensor whose shape will be used for padding size computation.

        Returns
        -------
        Tensor
        """
        return pad_as(x, y)


class MulModule(Module):

    def __new__(cls, input_shape, module, num_repeats, **kwargs):
        modules = []
        shape = input_shape
        for i in range(num_repeats):
            x = module(input_shape=shape, **kwargs)
            modules.append(x)
            shape = x.output_shape
        return Sequential(*modules)


class OrModule(Module):

    def __new__(cls, input_shape, modules):
        branches = []
        for module in modules:
            x = module(input_shape=input_shape)
            if isinstance(x, Branches) and x.mode is None:
                x = list(x.branches)
            else:
                x = [x]
            branches.extend(x)
        return Branches(branches, mode=None)


class Identity(Module):

    def __init__(self, input_shape, **kwargs):
        """ Identity mapping layer.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that
            batch dimension is not taken into account.
        """
        super().__init__(input_shape, **kwargs)

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for identity layer.

        Parameters
        ----------
        inputs : Tensor
            input tensor.

        Returns
        -------
        Tensor
            flattened input tensor.
        """
        return inputs


class Encoders(Module):

    def __init__(self, *modules):
        input_shapes = np.stack([m.input_shape
                                 for m in modules], axis=0)
        output_shapes = np.stack([m.output_shape
                                  for m in modules], axis=0)
        for i in range(len(input_shapes) - 1):
            if np.any(output_shapes[i] != input_shapes[i + 1]):
                raise ValueError("Input shape of each next module "
                                 + "must be equal to output shape of previous one")
        super().__init__(input_shapes[0, :])
        self.layers = torch.nn.ModuleList(modules)

    @property
    def output_shape(self):
        output_shapes = np.stack([m.output_shape
                                  for m in self.layers],
                                 axis=0)
        return output_shapes

    def forward(self, inputs):
        outputs = []
        x = inputs
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs


class Sum(Module):

    def __init__(self, encoder):
        out_shape = encoder.output_shape[0]
        if np.any(encoder.output_shape != out_shape):
            raise ValueError("Encoder outputs must"
                             + " have equal output shapes")
        super().__init__(encoder.input_shape)
        self.layers = encoder

    @property
    def output_shape(self):
        return self.layers.output_shape[0]

    def forward(self, inputs):
        output = 0
        for x in self.layers(inputs):
            output = output + x
        return output


class Prod(Module):

    def __init__(self, encoder):
        out_shape = encoder.output_shape[0]
        if np.any(encoder.output_shape != out_shape):
            raise ValueError("Encoder outputs must"
                             + " have equal output shapes")
        super().__init__(encoder.input_shape)
        self.layers = encoder

    @property
    def output_shape(self):
        return self.layers.output_shape[0]

    def forward(self, inputs):
        output = 1
        for x in self.layers(inputs):
            output = output * x
        return output


class Cat(Module):

    def __init__(self, encoder):
        out_shape = encoder.output_shape[0]
        if np.any(encoder.output_shape[:, 1:] != out_shape[1:]):
            raise ValueError("Encoder outputs must have equal "
                             + " output shapes along spatial dimensions")
        super().__init__(encoder.input_shape)
        self.layers = encoder

    @property
    def output_shape(self):
        shape = np.array([0, 0, 0])
        shape[0] = np.sum(self.layers.output_shape[:, 0])
        shape[1:] = self.layers.output_shape[0, 1:]
        return shape

    def forward(self, inputs):
        return torch.cat(self.layers(inputs), dim=1)


class Stack(Module):

    def __init__(self, *modules):
        input_shapes = np.stack([module.input_shape
                                 for module in modules],
                                axis=0)
        super().__init__(input_shapes)
        self.layers = torch.nn.ModuleList(modules)

    @property
    def output_shape(self):
        return np.stack([module.output_shape
                         for module in self.layers],
                        axis=0)

    def forward(self, inputs):
        return [l(x) for l, x in zip(self.layers,
                                     inputs)]


class Pyramid(Module):

    def __init__(self, encoders, decoders, mode='+'):
        if mode not in '+*':
            raise ValueError("Argument 'mode' must be one of"
                             + " following values: '+.*'.")
        for encoder, decoder in zip(encoders[::-1], decoders):
            if np.any(encoder.output_shape != decoder.input_shape):
                raise ValueError("Output shapes of encoders does not match"
                                 " input shapes of decoders")
        input_shapes = np.stack([encoder.input_shape
                                 for encoder in encoders], axis=0)
        super().__init__(input_shapes)
        self.encoders = torch.nn.ModuleList(encoders)
        self.decoders = torch.nn.ModuleList(decoders)
        self.mode = str(mode)

    @property
    def output_shape(self):
        shape = [self.encoders[-1].output_shape]
        shape += [decoder.output_shape
                  for decoder in self.decoders]
        return np.stack(shape, axis=0)

    def forward(self, inputs):
        encoded = self.encoders(inputs)
        encoded, x = encoded[:-1], encoded[-1]
        outputs = [x]
        for decoder, y in zip(self.decoders,
                              encoded[::-1]):
            x = self.merge(decoder(x, y, how=self.mode))
            outputs.append(x)
        return outputs


class ArithmeticModule(Module):

    def __init__(self, input_shape, op, value=None):
        super().__init__(input_shape)
        allowed_values = ('log', 'exp', 'sqrt', 'abs',
                          'rdiv', 'ldiv', 'pow' 'clip', 'mul', 'add',
                          'lsub', 'rsub')
        if op not in allowed_values:
            raise ValueError("Argument 'op' must be one "
                             + "of following values: {}".format(allowed_values))

        if isinstance(value, LIST_TYPES):
            value = np.array(value, np.float)
            shape = np.array(value.shape)
            if np.any(self.input_shape[-len(shape):] != shape):
                raise ValueError("Can't broadcast value to shape of input")
            value = torch.nn.Parameter(torch.from_numpy(value),
                                       requires_grad=False)
        elif isinstance(value, FLOAT_TYPES + INT_TYPES):
            value = float(value)
        else:
            value = None

        if op == 'log':
            self.op = torch.log
            self.value = None
            self.register_parameter('value', None)
        elif op == 'exp':
            self.op = torch.exp
            self.value = None
        elif op == 'sqrt':
            self.op = torch.sqrt
            self.value = None
            self.register_parameter('value', None)
        elif op == 'abs':
            self.op = torch.abs
            self.value = None
            self.register_parameter('value', None)
        elif op == 'ldiv' or op == 'rdiv':
            self.op = torch.div
            self.value = value
        elif op == 'pow':
            self.op = torch.pow
            self.value = value
        elif op == 'mul':
            self.op = torch.mul
            self.value = value


class Branches(Module):

    @classmethod
    def check_broadcast_shapes(cls,
                               shapes: list,
                               mode: str = '.',
                               message: str = ''):
        shapes = np.stack(shapes, axis=0)
        if mode == '.':
            if not np.all((shapes[:, 1:] == shapes[:, 1:].max(axis=0)) | (shapes[:, 1:] == 1)):
                raise ValueError(message)
            return np.concatenate([
                shapes[:, 0, None],
                np.repeat(np.max(shapes[None, :, 1:], axis=1),
                          repeats=shapes.shape[0], axis=0)], axis=1)
        else:
            if not np.all((shapes == shapes.max(axis=0)) | (shapes == 1)):
                raise ValueError(message)
            return np.repeat(np.max(shapes[None, ...], axis=1),
                             repeats=shapes.shape[0], axis=0)

    def __init__(self, branches: list, mode='.'):

        if mode not in ('.', '+', '*', None):
            raise ValueError("Invalid mode for branches."
                             + " Must be one of ('+', '.', '*', None)."
                             + " Got '{}' instead.".format(mode))

        input_shapes = np.stack([branch.input_shape
                                 for branch in branches], axis=0)
        if np.any(input_shapes != input_shapes[0, :]):
            raise ValueError("All branches must have same input shape")

        input_strides = np.stack([branch.stride
                                  for branch in branches
                                  if branch.stride is not None], axis=0)

        _ndims = max([branch.ndims for branch in branches])
        output_shapes = [
            np.concatenate([
                np.atleast_2d(branch.output_shape),
                branch.num_outputs * [[1] * (
                    _ndims - np.atleast_2d(branch.output_shape).shape[1])
                ]
            ], axis=1)
            for branch in branches
        ]
        output_shapes = np.concatenate(output_shapes, axis=0)
        if mode is None:
            self._output_shape = output_shapes
        elif mode == '.':
            message = ("All branches must have same output shape"
                       + " along spatial dimensions"
                       + " if 'mode' argument is set to '.'")
            output_shapes = self.check_broadcast_shapes(shapes=output_shapes,
                                                        mode=mode, message=message)
            channels = np.sum(output_shapes[:, 0])
            self._output_shape = np.array([channels,
                                           *output_shapes[0, 1:]])
        else:
            message = ("All branches must have same output shape"
                       " if 'mode' argument is set to '{mode}'")
            output_shapes = self.check_broadcast_shapes(shapes=output_shapes,
                                                        mode=mode, message=message)
            self._output_shape = output_shapes[0, :]

        if mode is not None:
            # TODO Remove commented code here and add normal text hint
            # if np.any(input_strides != input_strides[0, :]):
            #     raise ValueError("All branches must have same stride")
            self._stride = np.min(input_strides, axis=0)
        else:
            self._stride = input_strides

        super(Branches, self).__init__(input_shapes[0, :])
        self.branches = torch.nn.ModuleList(list(branches))
        self.mode = mode

    @property
    def stride(self) -> 'Tuple[float]':
        """ Get stride associated with layer.

        Returns
        -------
        Tuple[float] or None
            tuple of size ndims - 1 or None if stride
            does not exist for current layer.
        """
        if self.mode is None:
            return tuple(tuple(row) for row in self._stride.tolist())
        else:
            return tuple(self._stride.tolist())

    @property
    def output_shape(self):
        """ Get shape of the output tensor.

        Returns
        -------
        ndarray(int)
            shape of the output tensor.
        """
        return self._output_shape

    @staticmethod
    def _broadcast(a, b):
        if a.ndim == b.ndim:
            return a, b
        a_ = a.view(*[
            list(map(int, a.size()))
            + [1] * (max(a.ndim, b.ndim) - a.ndim)
        ])
        b_ = b.view(*[
            list(map(int, b.size()))
            + [1] * (max(a.ndim, b.ndim) - b.ndim)
        ])
        return a_, b_

    def forward(self, x: 'Tensor') -> 'Tensor':
        """ Forward pass method for parallel branches module block.

        Parameters
        ----------
        x : Tensor
            input tensor.

        Returns
        -------
        Tensor
            output tensor that is result of merge operation on
            outputs from different branches.
        """
        outputs = [branch(x) for branch in self.branches]
        if self.mode == '.':
            return torch.cat([u.expand(u.size(0), *self.output_shape.astype(int))
                              for u in outputs], dim=1)
        elif self.mode == '+':
            z = outputs[0]
            for y in outputs[1:]:
                z, y = self._broadcast(z, y)
                z = z + y
            return z
        elif self.mode == '*':
            z = outputs[0]
            for y in outputs[1:]:
                z, y = self._broadcast(z, y)
                z = z * y
            return z
        return outputs


class Layer(Module):

    def forward(self, inputs):
        return self.layer.forward(inputs)

    def __repr__(self):
        return self.layer.__repr__()


class Sequential(torch.nn.Sequential, metaclass=MetaModule):
    """ Base class for Sequential models. """

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def input_shape(self) -> 'ndarray':
        """ Get shape of the input tensor. """
        first, *_ = list(self.children())
        return first.input_shape

    @property
    def output_shape(self) -> 'ndarray':
        """ Get shape of the output tensor. """
        *_, last = list(self.children())
        return last.output_shape

    @property
    def stride(self) -> 'Tuple[float]':
        """ Get stride associated with layer.

        Returns
        -------
        Tuple[float] or None
            tuple of size ndims - 1 or None if stride
            does not exist for current layer.
        """
        stride = np.ones(self.ndims - 1,
                         dtype=np.float)
        for module in self.children():
            if module.stride is None:
                return None
            else:
                stride *= np.array(module.stride)
        return tuple(float(s) for s in stride)

    @property
    def ndims(self) -> int:
        """ Number of dimensions in input tensor.

        Batch dimension is not taken into account.
        """
        if self.input_shape.ndim == 2:
            return int(self.input_shape.shape[1])
        return len(self.input_shape)

    @property
    def in_channels(self):
        """ Get number of channels in the input tensor of operation.

        Returns
        -------
        int or ndarray(int)
            size of channels dimension of the input tensor.
        """
        if self.input_shape.ndim == 2:
            return self.input_shape[:, 0]
        return int(self.input_shape[0])

    @property
    def out_channels(self):
        """ Get number of channels in the output tensor of operation.

        Returns
        -------
        int or ndarray(int)
            size of channels dimension in the output tensor.
        """
        if self.output_shape.ndim == 2:
            return self.output_shape[:, 0]
        return int(self.output_shape[0])

    @property
    def num_inputs(self) -> int:
        """ Get number of input tensors for module.

        Returns
        -------
        int
            number of output tensors.
        """
        if self.input_shape.ndim == 1:
            return 1
        return self.input_shape.shape[0]

    @property
    def num_outputs(self) -> int:
        """ Get number of output tensors for module.

        Returns
        -------
        int
            number of output tensors.
        """
        if self.output_shape.ndim == 1:
            return 1
        return self.output_shape.shape[0]

    def to_int_array(self, parameter, name, length):
        """ Transform input parameter value to int list of given length.

        Parameters
        ----------
        parameter : int, tuple(int), list(int) or ndarray(int)
            input parameter value.
        name : str
            name of parameter. Required by exception raising part of function.
        length : int
            length of output list with parameter values.

        Returns
        -------
        ndarray(int)

        Raises
        ------
        ValueError
            If input parameter has wrong type or has improper length(if list-like).
        """
        if isinstance(parameter, INT_TYPES):
            parameter = np.asarray([parameter] * length, dtype=np.int)
        elif isinstance(parameter, LIST_TYPES):
            parameter = np.asarray(parameter, dtype=np.int).flatten()
            if len(parameter) != length:
                raise ValueError("Argument {} has inproper lenght.".format(name)
                                 + " Must have {}, got {}.".format(length,
                                                                   len(parameter)))
        else:
            raise ValueError("Argument {} must be int or ".format(name)
                             + "tuple, list, ndarray "
                             + "containing {} int values.".format(length))
        return parameter

    @classmethod
    def merge(cls, x, y, how='+'):
        """ Merge tensors according given rule.

        Parameters
        ----------
        x : Tensor
            first tensor.
        y : Tensor
            second tensor.
        how : str
            how to merge input tensors. Can be on of following values:
            '+' for sum, '*' for product or '.' for concatenation along first
            dimension. Default is '+'.

        Returns
        -------
        Tensor
            result of merging operation.

        Raises
        ------
        ValueError
            if argument 'how' has value diverging from '+', '*' or '.'.
        """
        if how not in ('+', '*', '.'):
            raise ValueError("Argument 'how' must be one of "
                             + "following values: ('+', '.', '*'). "
                             + "Got {}.".format(how))
        return merge([x, y], how)

    @classmethod
    def crop_as(cls, x: 'Tensor', y: 'Tensor') -> 'Tensor':
        """ Crop first tensor to have the same shape as the second.

        This method affects only spatial dimensions of tensor, so
        batch size and channels dimension remain unchanged.

        Parameters
        ----------
        x : Tensor
            tensor to crop.
        y : Tensor
            tensor whose shape will be used for cropping.

        Returns
        -------
        Tensor
        """
        return crop_as(x, y)

    @classmethod
    def pad_as(cls, x: 'Tensor', y: 'Tensor') -> 'Tensor':
        """ Add padding to first tensor to have the same shape as the second.

        This method affects only spatial dimensions of tensor, so
        batch size and channels dimension remain unchanged.


        Parameters
        ----------
        x : Tensor
            tensor to pad.
        y : Tensor
            tensor whose shape will be used for padding size computation.

        Returns
        -------
        Tensor
        """
        return pad_as(x, y)


class ConvModule(Module):
    """ Base class for all Convolutional modules.

    Following layers are considered convolutional:
    Conv1d, Conv2d, Conv3d, ConvTranspose1d,
    ConvTranspose2d, ConvTranspose3d,
    MaxPool1d, MaxPool2d, MaxPool3d, MaxUnpool.
    """
    _repr_attributes = []

    @property
    def output_shape(self):
        """ Get shape of the output tensor.

        Returns
        -------
        tuple(int)
            shape of the output tensor.
        """
        return self._output_shape

    @property
    def stride(self) -> 'Tuple[float]':
        """ Get stride associated with layer.

        Returns
        -------
        Tuple[float] or None
            tuple of size ndims - 1 or None if stride
            does not exist for current layer.
        """
        return self._stride[:]

    @property
    def kernel_size(self) -> 'Tuple[int]':
        """ Get kernel size associated with layer.

        Returns
        -------
        Tuple[int]
            tuple of size ndims - 1 representing
            kernel size along spatial axes.
        """
        return self._kernel_size[:]

    @property
    def dilation(self) -> 'Tuple[int]':
        """ Get dilation rate associated with layer.

        Returns
        -------
        Tuple[int]
            tuple of size ndims - 1 representing
            dilation rate along spatial axes.
        """
        return self._dilation[:]

    def __init__(self, input_shape, kernel_size=3, stride=1, dilation=1):
        super().__init__(input_shape)
        if self.ndims not in (2, 3, 4):
            raise ValueError("Input tensor must be 2, 3 or 4 dimensional "
                             + " with zero axis meaning number of channels.")
        self._kernel_size = transform_to_int_tuple(kernel_size,
                                                   'kernel_size',
                                                   self.ndims - 1)
        self._stride = transform_to_int_tuple(stride,
                                              'stride',
                                              self.ndims - 1)
        self._dilation = transform_to_int_tuple(dilation,
                                                'dilation',
                                                self.ndims - 1)
