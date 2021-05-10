""" Contains ConvBlock, Branches, NoOperation and Merge classes. """

import inspect
import copy
from functools import reduce
import operator
from collections import Counter, OrderedDict
from collections import defaultdict
import re
import numpy as np
import torch

from ..bases import *
from ..utils import *


class ConvBlock(Sequential, metaclass=MetaModule):

    @classmethod
    def get_options(cls) -> dict:
        """ Get registered options.

        Returns
        -------
        dict
            dictionary with options: keys are shortcuts, values are corresponding
            modules' classes.
        """
        return cls._options.copy()

    @classmethod
    def _unify_parameter(cls, values, vectorize, num_layers, ndims):
        """ Unify parameter values passed to ConvBlock. """

        if isinstance(values, LIST_TYPES):

            # This is a kind of a hook: if there is only one layer of this type
            # you won't need to write smth like
            # c=dict(kernel_size=[(1, 3)]) if layout='c'
            if num_layers == 1 and ((len(values) == ndims and vectorize) or not vectorize):
                if len(values) == 1:
                    return values
                else:
                    return [values]

            # Here is just a regular vectorized logics
            if len(values) != num_layers:
                raise ValueError("Length of param {}".format(len(values))
                                 + " must match number"
                                 + " of layers in layout"
                                 + " which is {}".format(num_layers))

            if all(isinstance(v, INT_TYPES + FLOAT_TYPES) for v in values) and vectorize:
                return [(v, ) * ndims for v in values]
            else:
                return values

        if isinstance(values, bool):
            values = [values] * num_layers
        elif isinstance(values, INT_TYPES):
            values = [int(values)] * num_layers
        elif isinstance(values, FLOAT_TYPES):
            values = [float(values)] * num_layers
        else:
            values = [values] * num_layers

        if vectorize:
            return [(value, ) * ndims for value in values]

        return values

    @classmethod
    def register_option(cls, name, vectorized_params=()):
        """ Decorator used to register options for ConvBlock.

        Parameters
        ----------
        name : str
            name of shortcut for registered option.
            For example, 'c' for Conv layer.
        vectorized_params : Tuple[str]
            names of arguments for option that will be
            expanded depending on number of dimensions.

        Returns
        -------
        Callable
            decorator for module class.
        """

        def decorator(module_cls):
            cls._options[name] = module_cls

            params_description = []

            if isinstance(module_cls, type) and type(module_cls) != type:
                params_info = inspect.getfullargspec(module_cls.__init__)
                args_names = params_info.args[2:]
            else:
                params_info = inspect.getfullargspec(module_cls)
                args_names = params_info.args[1:]

            if params_info.defaults:
                args_defaults = dict(zip(args_names[::-1],
                                         params_info.defaults[::-1]))
            else:
                args_defaults = {}
            for i in range(len(args_names)):
                arg_name = args_names[i]
                param_dict = {'name': arg_name}
                if arg_name in vectorized_params:
                    param_dict['vectorize'] = True
                else:
                    param_dict['vectorize'] = False

                if arg_name in args_defaults:
                    param_dict['default'] = args_defaults[arg_name]

                params_description.append(param_dict)

            cls._options_params[name] = params_description
            return module_cls

        return decorator

    @classmethod
    def map_layout_to_options(cls, layout):
        return list(re.sub(r'\s+', '', layout))

    def __init__(self, input_shape, layout, **kwargs):
        """ Create convolutional block module.

        Covnolutional block is a subclass of pytorch sequential model.
        User can dynamically register custom pytorch modules
        as options of ConvBlock and use shortcuts corresponding
        to these modules in layout of ConvBlock using
        ConvBlock.register_option decorator. Calling get_options
        classmethod will return all registered options that can be used in
        ConvBlock layout.

        Many layers contain vectorized parameters like 'kernel_size',
        'stride' or 'dilation' that means that these parameters can have
        different values for different dimensions. Also each created block
        can contain several operations of given type(in our case
        two convolutions, two activations and two batch normalizations).
        Usually it is required to pass the same value for specific parameter
        for each dimension or/and for all operations of given type in block.
        ConvBlock gives an ability to avoid explicit copying of parameters values
        for similar operations in block. For example, c=dict(kernel_size=3)
        in ConvBlock will mean that all convolutions will
        have (3, 3) kernel size (in 2D case). If one wants to pass different
        kernel_size parameter for each convolution then it's possible to do that
        passing list of parameters c=dict(kernel_size=[3, 5]). Note that in this
        case length of list or tuple must be the same as number of
        corresponding operations in block. If it's required
        to have non-symmetric kernel_size for convolutions along xy-dims
        but shared by all operations in block then passing
        c=dict(kernel_size=(3, 5)) will solve the problem.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that
            batch dimension is not taken in account.
        layout : str
            compressed description of the block
            operations sequence using shortcuts for registered options.
            For instance, 'cna cna' will represent two convolution operations
            with batch normalization before activation.
        **kwargs
            each argument must be a dict with keys representing parameters of
            corresponding operation(For example, c=dict(kernel_size=3, filters=16))

        Note
        ----
        This module also have partial(...) method that allows to split
        parameters passing for __init__ or __new__
        constructor into several steps.


        Examples
        --------
        Creation of block of two convolutions with batch normalization before
        activation followed by max pooling operation will look like:

        >>> x = ConvBlock(
        ... input_shape=(3, 128, 128), layout='cna cna p',
        ... c=dict(kernel_size=3, filters=(16, 32)),
        ... p=dict(kernel_size=2, stride=2)
        ... a=dict(activation=)
        ... )

        # TODO: Add 'Raises' section in docstring
        """
        super().__init__()

        self.name = kwargs.get('name', self.__class__.__name__)
        self.layout = self.map_layout_to_options(layout)
        self.layers_counter = Counter(self.layout)

        assert self.layers_counter.get(
            '(', 0) == self.layers_counter.get(')', 0)
        assert self.layers_counter.get(
            '+', 0) == self.layers_counter.get('+', 0)
        assert self.layers_counter.get(
            '*', 0) == self.layers_counter.get('*', 0)
        assert self.layers_counter.get(
            '.', 0) == self.layers_counter.get('.', 0)

        input_shape = self.to_int_array(input_shape,
                                        'input_shape',
                                        len(input_shape))
        ndims = len(input_shape)
        layers_params = {layer_name: {} for layer_name in self.layers_counter}
        for layer_name, layer_counts in self.layers_counter.items():
            layer_kwargs = kwargs.get(layer_name, {})
            for param in self._options_params[layer_name]:
                if param['name'] in layer_kwargs:
                    raw_value = layer_kwargs[param['name']]
                elif 'default' in param:
                    raw_value = param['default']
                else:
                    raise ValueError("Argument {} ".format(param['name'])
                                     + "has no default value")

                values = self._unify_parameter(raw_value, param['vectorize'],
                                               layer_counts, ndims - 1)
                layers_params[layer_name][param['name']] = values

        # self.layers_params = copy.deepcopy(layers_params)
        shape = transform_to_int_tuple(input_shape, 'input_shape', ndims)
        for i, layer in enumerate(self.layout):
            layer_class = self._options[layer]
            params_dict = {}
            for param in self._options_params[layer]:
                param_values = layers_params[layer][param['name']]
                params_dict[param['name']] = param_values[0]
                layers_params[layer][param['name']] = param_values[1:]

            module = layer_class(shape, **params_dict)
            self.add_module('Module_{}'.format(i), module)

            shape = transform_to_int_tuple(module.output_shape,
                                           'output_shape',
                                           len(module.output_shape))

        self._input_shape = np.array(input_shape, dtype=np.int)

    @property
    def input_shape(self) -> 'ndarray(int)':
        """ Get shape of the input tensor.

        Returns
        -------
        ndarray(int)
            shape of the input tensor (1d array)
        """
        return np.array(self._input_shape, dtype=np.int)

    def __repr__(self) -> str:
        """ String representation of ConvBlock. """
        tmpstr = self.name + '(\n'
        for key, module in self._modules.items():
            modstr = module.__repr__()
            modstr = addindent(modstr, 2)
            tmpstr = tmpstr + '  (' + key + '): ' + modstr + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr


class ResConvBlock(ConvBlock):

    _options = ConvBlock._options
    _options_params = ConvBlock._options_params

    @classmethod
    def unify_layer_params(cls,
                           layer_name: str,
                           layer_counts: int,
                           layer_kwargs: dict, ndim: int,
                           allow_missing: bool = False) -> dict:
        out_params = {}
        for param in cls._options_params[layer_name]:
            if param['name'] in layer_kwargs:
                raw_values = layer_kwargs[param['name']]
            elif 'default' in param:
                raw_values = param['default']
            elif not allow_missing:
                raise ValueError("Argument {} ".format(param['name'])
                                 + "has no default value")
            else:
                continue
            out_params[param['name']] = cls._unify_parameter(raw_values,
                                                             param['vectorize'],
                                                             layer_counts, ndim)
        return out_params

    @classmethod
    def map_layout_to_options(cls, layout):
        return list(re.sub(r'\s+', '', layout))

    @classmethod
    def split_params(cls, params, layout):
        kwargs = {layer: defaultdict(list)
                  for layer in layout}
        for layer in layout:
            for param_name, param_value in params[layer].items():
                x = param_value.pop(0)
                kwargs[layer][param_name].append(x)
        return kwargs

    @classmethod
    def build_block(cls, input_shape,
                    layout, layers_params):
        i = 0
        start = 0
        mode = None
        layers = []
        params = layers_params
        shape = input_shape
        shortcut_params = params.get('shortcut', {})
        shortcut_creator = cls._options['shortcut']
        magic_seq = []
        while i < len(layout):
            if layout[i] in '+.*/' and mode is None:
                if i - start > 0:
                    kwargs = cls.split_params(params, layout[start: i])
                    module = ConvBlock(input_shape=shape,
                                       layout=''.join(layout[start: i]),
                                       **kwargs)
                    layers.append(module)
                    if layout[i] == '/':
                        magic_seq.append('/')
                    else:
                        magic_seq.append('-')
                    shape = module.output_shape
                elif i - start == 0 and layout[i] == '/':
                    magic_seq[-1] = '/'

                start = i + 1
                mode = layout[i] if layout[i] != '/' else None
            elif layout[i] == mode:
                kwargs = cls.split_params(params, [l for l in layout[start: i]
                                                   if l not in '+.*'])
                if any(c in layout[start: i] for c in '+.*'):
                    module = cls.build_block(shape,
                                             ''.join(layout[start: i]),
                                             kwargs)
                else:
                    module = ConvBlock(
                        input_shape=shape,
                        layout=''.join(layout[start: i]),
                        **kwargs
                    )

                shortcut_kwargs = {name: value.pop(0)
                                   for name, value
                                   in shortcut_params.items()}
                shortcut_kwargs.update({'stride': module.stride,
                                        'mode': mode})
                shortcut = shortcut_creator(
                    input_shape=module.input_shape,
                    output_shape=module.output_shape,
                    **shortcut_kwargs
                )
                module = Branches([
                    module,
                    shortcut
                ], mode=mode)

                layers.append(module)
                magic_seq.append('-')
                shape = module.output_shape
                start = i + 1
                mode = None
            elif layout[i] == '/':
                raise ValueError("Split inside of residual"
                                 + " block is not allowed")

            i += 1

        if (i == len(layout)
            and i - start > 0
                and mode is None):

            kwargs = cls.split_params(params, layout[start: i])
            module = ConvBlock(
                input_shape=shape,
                layout=''.join(layout[start: i]),
                **kwargs
            )
            layers.append(module)
            magic_seq.append('-')

        blocks = []
        block_layers = []
        for layer, magic in zip(layers, magic_seq):
            if magic == '-':
                block_layers.append(layer)
            elif magic == '/':
                blocks.append(Sequential(*block_layers, layer))
                block_layers = []
            else:
                raise ValueError("Unknown magic sequence"
                                 + " value: '{}'".format(magic))
        if len(block_layers) > 0:
            blocks.append(Sequential(*block_layers))
        if len(blocks) == 1:
            return blocks[0]
        return Encoders(*blocks)

    def __new__(cls, input_shape, layout, **kwargs):

        num_res = 0
        num_res += max(layout.count('+') - 1, 0)
        num_res += max(layout.count('.') - 1, 0)
        num_res += max(layout.count('*') - 1, 0)

        num_splits = layout.count('/')
        if num_res == 0 and num_splits == 0:
            return ConvBlock(input_shape=input_shape,
                             layout=layout, **kwargs)

        layout = cls.map_layout_to_options(layout)
        layers_counter = {**Counter(layout),
                          'shortcut': num_res}

        if (layers_counter.pop('+', 0) % 2 > 0
                or layers_counter.pop('.', 0) % 2 > 0
                or layers_counter.pop('*', 0) % 2 > 0):
            raise ValueError("Number of residual symbols must be even")

        _ = layers_counter.pop('/', 0)

        input_shape = transform_to_int_tuple(input_shape,
                                             'input_shape',
                                             len(input_shape))

        layers_params = {}
        for layer_name, layer_counts in layers_counter.items():
            if layer_name == 'shortcut':
                allow_missing = True
            else:
                allow_missing = False
            values = cls.unify_layer_params(layer_name, layer_counts,
                                            kwargs.get(layer_name, {}),
                                            len(input_shape) - 1,
                                            allow_missing=allow_missing)
            layers_params[layer_name] = values

        return cls.build_block(input_shape, layout, layers_params)


@ResConvBlock.register_option(name='shortcut', vectorized_params=[
    'kernel_size', 'stride', 'pool_size', 'pool_stride'])
def res_shortcut(input_shape,
                 output_shape,
                 layout='cna',
                 kernel_size=1,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=False,
                 pool_size=2,
                 pool_mode='max',
                 allow_identity=True,
                 broadcast=True,
                 mode='+',
                 filters=None,
                 downsample_mode='c',
                 **kwargs):
    assert downsample_mode in 'cp'
    ndims = len(input_shape) - 1
    if stride is None:
        stride = 1
    stride = np.array(transform_to_int_tuple(stride,
                                             'stride',
                                             ndims))
    if mode in '+*' and filters is not None:
        raise ValueError(
            "Argument 'filters' must be None if mode is '+' or '*'")
    elif filters is None:
        filters = int(output_shape[0])
    else:
        filters = int(filters)

    if (allow_identity and np.all(stride == 1)):
        if (mode == '.'
            and np.all(input_shape[1:]
                       == output_shape[1:])):
            return Identity(input_shape=input_shape)
        elif np.all(input_shape == output_shape):
            return Identity(input_shape=input_shape)
    elif allow_identity:
        cond = True
        for q, s in zip(output_shape[1:], stride):
            if (s != 1 and q == 1) or s == 1:
                cond &= True
            else:
                cond &= False
        if cond:
            return Identity(input_shape=input_shape)
            
    if (downsample_mode == 'p') or ('c' not in layout):
        pool_stride = [stride.tolist()] + min(layout.count('p') - 1, 0) * [1]
        conv_stride = 1
    elif downsample_mode == 'c':
        pool_stride = 1
        conv_stride = [stride.tolist()] + min(layout.count('c') - 1, 0) * [1]
    return ConvBlock(
        input_shape=input_shape,
        layout=layout,
        c=dict(filters=output_shape[0],
               kernel_size=kernel_size,
               stride=conv_stride,
               dilation=dilation,
               groups=groups,
               bias=bias),
        p=dict(kernel_size=pool_size,
               stride=pool_stride,
               mode=pool_mode),
        **kwargs
    )


class ConvBranches(Module):

    def __new__(cls, input_shape, mode='.', **kwargs):
        if len(kwargs) == 0:
            raise ValueError("Configuration for at least"
                             + " one branch must be provided")
        elif len(kwargs) == 1:
            names = list(kwargs.keys())
            return ResConvBlock(input_shape=input_shape,
                                **kwargs[names[0]])
        return super(ConvBranches, cls).__new__(cls)

    def __init__(self, input_shape, mode='.', **kwargs):
        super().__init__(input_shape)
        if mode not in ('.', '+', '*', None):
            raise ValueError("Invalid mode for branches."
                             + " Must be one of ('+', '.', '*', None)."
                             + " Got '{}' instead.".format(mode))
        self.mode = mode
        self.branches = torch.nn.ModuleDict({
            name: (ResConvBlock(input_shape=self.input_shape, **config)
                   if config is not None else Identity(input_shape=self.input_shape))
            for name, config in kwargs.items()
        })

        input_strides = np.stack([branch.stride
                                  for branch in self.branches.values()], axis=0)

        output_shapes = np.stack([branch.output_shape
                                  for branch in self.branches.values()], axis=0)
        if mode is None:
            self._output_shape = output_shapes
        elif mode == '.':
            if np.any(output_shapes[:, 1:] != output_shapes[0, 1:]):
                raise ValueError("All branches must have same output shape"
                                 + " along spatial dimensions"
                                 + " if 'mode' argument is set to '.'.")
            channels = np.sum(output_shapes[:, 0])
            self._output_shape = np.array([channels,
                                           *output_shapes[0, 1:]])
        else:
            if np.any(output_shapes != output_shapes[0, :]):
                raise ValueError("All branches must have same output shape"
                                 + " if 'mode' argument is set to '{}'.".format(mode))
            self._output_shape = output_shapes[0, :]

        if mode is not None:
            if np.any(input_strides != input_strides[0]):
                raise ValueError("All branches must have same stride")
            self._stride = input_strides[0]
        else:
            self._stride = input_strides

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
        outputs = []
        for branch in self.branches.values():
            v = branch(x)
            if len(v.size()) - 1 < self.ndims:
                shape = list(map(int, v.size()))
                shape += [1] * (1 + self.ndims - len(v.size()))
                v = v.view(*shape)
            outputs.append(v)

        if self.mode == '.':
            return torch.cat(outputs, 1)
        elif self.mode == '+':
            z = outputs[0]
            for y in outputs[1:]:
                z = z + y
            return z
        elif self.mode == '*':
            z = outputs[0]
            for y in outputs[1:]:
                z = z * y
            return z
        return outputs