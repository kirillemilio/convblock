import numpy as np
import torch
from ..layers import ConvBlock
from ..utils import transform_to_int_tuple
from ..config import Config
from ..blocks import RFB
from ..blocks import RFBa


class Pyramid(torch.nn.Module):

    @classmethod
    def default_config(cls):
        config = Config()
        config['downsample'] = {
            'layout': 'c',
            'c': {
                'kernel_size': 3,
                'stride': 2,
                'bias': True
            },
            'p': {
                'kernel_size': 2,
                'stride': 2,
                'mode': 'max'
            }
        }
        config['upsample'] = {
            'layout': 'u',
            't': {
                'kernel_size': 2,
                'stride': 2,
                'bias': True
            },
            'u': {
                'scale': 2,
                'mode': 'nearest'
            }
        }

        return config

    def pop_levels(self, config, num_levels=None):
        num_levels = -1 if num_levels is None else abs(int(num_levels))
        levels = config.pop('levels', {})
        if len(levels) > 0 and num_levels >= 0 and len(levels) != num_levels:
            raise ValueError("Number of levels in config"
                             + " must match required number of levels")
        elif len(levels) == 0 and num_levels >= 0:
            return {i: {} for i in range(num_levels)}
        elif len(levels) > 0 and num_levels is None:
            raise ValueError("Config must not contain 'levels'")
        return {i: levels.get(i, {}) for i in range(num_levels)}

    def build_config(self, config, **kwargs):
        return config

    def __init__(self, input_shape, config: dict = None, **kwargs):
        super(Pyramid, self).__init__()
        self.input_shape = np.array(input_shape)
        self.num_inputs = len(self.input_shape)


class FeaturesPyramid(torch.nn.Module):

    @classmethod
    def default_config(cls):
        return Pyramid.default_config() @ Config({
            'input': {
                'layout': 'c',
                'c': {
                    'kernel_size': 1,
                    'bias': True
                }
            },
            'output': {
                'layout': 'c',
                'c': {
                    'kernel_size': 3,
                    'bias': True
                }
            },
            'extra': {
                'layout': 'c',
                'c': {
                    'kernel_size': 3,
                    'bias': True,
                    'stride': 2
                }
            }
        })

    def __init__(self,
                 input_shape,
                 in_filters=256,
                 out_filters=256,
                 num_extra=2,
                 extra_filters=256,
                 config=None):
        super(FeaturesPyramid, self).__init__()

        self.input_shape = np.array(input_shape)
        num_inputs = len(self.input_shape)
        in_filters = transform_to_int_tuple(in_filters,
                                            'in_filters',
                                            num_inputs)
        out_filters = transform_to_int_tuple(out_filters,
                                             'out_filters',
                                             num_inputs)
        if num_extra == 0:
            extra_filters = []
        else:
            extra_filters = transform_to_int_tuple(extra_filters,
                                                   'extra_filters',
                                                   num_extra)

        config = self.default_config() @ Config(config)
        self.in_layers = torch.nn.ModuleList()
        self.up_layers = torch.nn.ModuleList()
        self.out_layers = torch.nn.ModuleList()
        self.extra_layers = torch.nn.ModuleList()
        for i in range(num_inputs):
            in_layer = ConvBlock(**(config['input']
                                    @ {'input_shape': self.input_shape[i, :],
                                       'c': {'filters': in_filters[i]}}))
            if i > 0:
                up_layer = ConvBlock(**(config['upsample']
                                        @ {'input_shape': in_layer.output_shape,
                                           't': {'filters': in_filters[i - 1]}}))
            else:
                up_layer = None

            out_layer = ConvBlock(**(config['output']
                                     @ {'input_shape': in_layer.output_shape,
                                        'c': {'filters': out_filters[i]}}))

            self.in_layers.append(in_layer)
            self.out_layers.append(out_layer)
            if up_layer is not None:
                self.up_layers.append(up_layer)

        shape = self.input_shape[-1, :]
        for i, f in enumerate(extra_filters):
            extra_layer = ConvBlock(**(config['extra']
                                       @ {'input_shape': shape,
                                          'c': {'filters': f}}))
            self.extra_layers.append(extra_layer)
            shape = extra_layer.output_shape

    @property
    def output_shape(self):
        shapes = [l.output_shape for l in self.out_layers]
        shapes += [l.output_shape for l in self.extra_layers]
        return np.array(shapes)

    def forward(self, inputs):
        outputs = []
        up_z = None
        for i, x in enumerate(inputs[::-1]):
            z = self.in_layers[-i-1](x)
            if up_z is not None:
                outputs.append(self.out_layers[-i-1](z + up_z))
            else:
                outputs.append(self.out_layers[-i-1](z))

            if i < len(self.in_layers) - 1:
                up_z = self.up_layers[-i-1](z)

        extra_outputs = []
        z = inputs[-1]
        for extra_layer in self.extra_layers:
            z = extra_layer(z)
            extra_outputs.append(z)

        return outputs[::-1] + extra_outputs


class RFBFeaturesPyramid(FeaturesPyramid):

    @classmethod
    def default_config(cls):
        return Pyramid.default_config() @ Config({
            'input': {
                'layout': 'c',
                'c': {
                    'kernel_size': 1,
                    'bias': True
                }
            },
            'output': {
                'layout': 'c',
                'c': {
                    'kernel_size': 3,
                    'bias': True
                }
            },
            'extra': {}
        })

    def __init__(self,
                 input_shape,
                 in_filters=256,
                 out_filters=256,
                 num_extra=2,
                 extra_filters=256,
                 config=None):
        super(FeaturesPyramid, self).__init__()

        self.input_shape = np.array(input_shape)
        num_inputs = len(self.input_shape)
        in_filters = transform_to_int_tuple(in_filters,
                                            'in_filters',
                                            num_inputs)
        out_filters = transform_to_int_tuple(out_filters,
                                             'out_filters',
                                             num_inputs)
        if num_extra == 0:
            extra_filters = []
        else:
            extra_filters = transform_to_int_tuple(extra_filters,
                                                   'extra_filters',
                                                   num_extra)

        config = self.default_config() @ Config(config)
        self.in_layers = torch.nn.ModuleList()
        self.up_layers = torch.nn.ModuleList()
        self.out_layers = torch.nn.ModuleList()
        self.extra_layers = torch.nn.ModuleList()
        for i in range(num_inputs):
            in_layer = ConvBlock(**(config['input']
                                    @ {'input_shape': self.input_shape[i, :],
                                       'c': {'filters': in_filters[i]}}))
            if i > 0:
                up_layer = ConvBlock(**(config['upsample']
                                        @ {'input_shape': in_layer.output_shape,
                                           't': {'filters': in_filters[i - 1]}}))
            else:
                up_layer = None

            out_layer = ConvBlock(**(config['output']
                                     @ {'input_shape': in_layer.output_shape,
                                        'c': {'filters': out_filters[i]}}))

            self.in_layers.append(in_layer)
            self.out_layers.append(out_layer)
            if up_layer is not None:
                self.up_layers.append(up_layer)

        shape = self.input_shape[-1, :]
        for i, f in enumerate(extra_filters):
            extra_layer = RFB(**(config['extra']
                                 @ {'input_shape': shape,
                                    'downsample': True,
                                    'out_filters': f}))
            self.extra_layers.append(extra_layer)
            shape = extra_layer.output_shape


class FusionPyramid(Pyramid):

    @classmethod
    def default_config(cls):
        return Pyramid.default_config() @ Config({
            'input': {
                'layout': 'c',
                'c': {
                    'kernel_size': 1,
                    'bias': True
                }
            },
            'output': {
                'layout': 'na cna',
                'c': {
                    'kernel_size': 3,
                    'bias': True,
                }
            },
        })

    def build_config(self, config, in_filters, out_filters, **kwargs):
        assert not any(c in config['upsample/layout'] for c in 'cpf')
        assert (config['upsample/layout'].count('t')
                + config['upsample/layout'].count('u') == 1)

        assert not any(c in config['downsample/layout'] for c in 'tu')
        assert (config['downsample/layout'].count('c')
                + config['downsample/layout'].count('p') == 1)
        return config @ {
            'input/levels': {i: {'c/filters': f}
                             for i, f in enumerate(in_filters)},
            'upsample/levels': {i: {'t/filters': f,
                                    't/stride': 2 ** (i + 1),
                                    'u/scale': 2 ** (i + 1)}
                                for i, f in enumerate(in_filters[1:])},
            'output/levels': {i: {'c/filters': f}
                              for i, f in enumerate(out_filters)},
            'downsample/levels': {i: {'c/filters': out_filters[i]}
                                  for i, f in enumerate(out_filters[1:])}
        }

    def __init__(self, input_shape,
                 in_filters=(256, 256, 256),
                 out_filters=(128, 256, 512),
                 mode: str = '.',
                 config: dict = None):

        super().__init__(input_shape, config)
        self.input_shape = np.array(input_shape)
        self.mode = str(mode) if mode else '.'
        config = self.build_config(self.default_config()
                                   @ Config(config),
                                   in_filters, out_filters)
        in_filters = np.array(in_filters, np.int).tolist()
        out_filters = np.array(out_filters, np.int).tolist()

        self.num_inputs = len(in_filters)
        self.num_outputs = len(out_filters)

        in_levels = self.pop_levels(config['input'],
                                    self.num_inputs)
        up_levels = self.pop_levels(config['upsample'],
                                    self.num_inputs - 1)
        in_layers = torch.nn.ModuleList()
        for i in range(self.num_inputs):
            x = ConvBlock(
                input_shape=self.input_shape[i, :],
                **(config['input'] @ in_levels[i])
            )
            if i > 0:
                ux = ConvBlock(
                    input_shape=x.output_shape,
                    **(config['upsample']
                       @ up_levels[i - 1])
                )
                x = Sequential(x, ux)
            in_layers.append(x)

        assert np.all(np.stack([l.output_shape[1:]
                                for l in in_layers], 0)
                      == self.input_shape[0, 1:])

        out_levels = self.pop_levels(config['output'],
                                     self.num_outputs)
        down_levels = self.pop_levels(config['downsample'],
                                      self.num_outputs - 1)
        if mode == '.':
            channels = sum(l.output_shape[0]
                           for l in in_layers)
        else:
            channels = in_layers[0].output_shape[0]
        shape = np.array([channels, *self.input_shape[0, 1:]])

        out_layers = torch.nn.ModuleList()
        for i in range(self.num_outputs):
            if i > 0:
                dx = ConvBlock(
                    input_shape=shape,
                    **(config['downsample']
                       @ down_levels[i - 1])
                )
                shape = dx.output_shape
            else:
                dx = None
            x = ConvBlock(
                input_shape=shape,
                **(config['output'] @ out_levels[i])
            )
            if dx is not None:
                x = Sequential(dx, x)
            out_layers.append(x)
            shape = x.output_shape

        self.in_layers = in_layers
        self.out_layers = out_layers

    @property
    def output_shape(self):
        return np.stack([layer.output_shape
                         for layer in self.out_layers], axis=0)

    def forward(self, inputs):
        ux = [] if self.mode == '.' else 0
        for x, in_layer in zip(inputs, self.in_layers):
            if self.mode == '.':
                ux.append(in_layer(x))
            elif self.mode == '+':
                ux = ux + in_layer(x)
            elif self.mode == '*':
                ux = ux * in_layer(x)

        ux = (torch.cat(ux, 1)
              if self.mode == '.'
              else ux)

        outputs = []
        x = ux
        for out_layer in self.out_layers:
            x = out_layer(x)
            outputs.append(x)
        return outputs
