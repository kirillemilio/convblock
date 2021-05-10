""" Contains implementation of parametrized ResNet model. """

from ..layers.conv_block import ResConvBlock as ConvBlock
from ..blocks import BottleneckResBlock
from ..blocks import SimpleResBlock
from ..bases import Sequential
from .base_model import BaseModel


class ResNet(BaseModel):

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()

        config['input'] = {
            'layout': 'cnap',
            'c': dict(kernel_size=7, stride=2, filters=64),
            'p': dict(kernel_size=3, stride=2)
        }

        config['body'] = {
            'filters': (64, 128, 256, 512),
            'shortcut': {'layout': 'cn'},
            'num_blocks': (3, 4, 6, 3),
            'use_bottleneck': False,
            'use_se': False,
            'reduction': 4,
            'factor': 4,
            'cardinality': 1
        }

        config['head'] = {
            'layout': '> fa',
            '>': dict(mode='avg'),
            'f': dict(out_features=10),
            'a': dict(activation='linear')
        }
        return config

    @classmethod
    def block(cls,
              input_shape,
              filters: int,
              factor: int = 4,
              reduction: int = 4,
              dilation: int = 1,
              cardinality: int = 1,
              downsample: int = 0,
              use_se: bool = False,
              use_bottleneck: bool = False,
              post_activation: bool = True,
              post_norm: bool = False,
              **kwargs):
        filters, factor, reduction = int(filters), int(factor), int(reduction)
        dilation, cardinality = int(dilation), int(cardinality)
        filters = (
            [filters]
            + use_bottleneck * [filters]
            + [round(filters * factor)]
            + use_se * [
                round(filters * factor / reduction),
                round(filters * factor)
            ]
        )
        kernel_size = ([1, 3, 1] if use_bottleneck else [3, 3]) + use_se * [1, 1]

        stride = [1] * (downsample - 1) + [2 if downsample else 1]
        if use_bottleneck:
            assert len(stride) <= 3
            stride += (3 - len(stride)) * [1]
        else:
            assert len(stride) <= 2
            stride += (2 - len(stride)) * [1]
        stride += use_se * [1, 1]

        groups = [1, cardinality] + use_bottleneck * [1] + use_se * [1, 1]
        dilation = [1, dilation] + use_bottleneck * [1] + use_se * [1, 1]
        bias = [False, False] + use_bottleneck * [False] + use_se * [True, True]
        activation = (['relu'] + use_bottleneck * ['relu']
                      + use_se * ['relu', 'sigmoid']
                      + post_activation * ['relu'])
        layout = ('+ cna' + use_bottleneck * 'cna'
                  + 'cn' + use_se * '* p ca ca *'
                  + '+' + post_norm * 'n'
                  + 'a' * post_activation)
        return ConvBlock(
            input_shape=input_shape, layout=layout,
            c=dict(kernel_size=kernel_size, stride=stride,
                   filters=filters, bias=bias,
                   dilation=dilation, groups=groups),
            p=dict(output_size=1, adaptive=True, mode='avg'),
            a=dict(activation=activation),
            **kwargs
        )

    def build_body(self, input_shape, config):
        use_bottleneck = config.get('use_bottleneck')
        use_se = config.get('use_se', False)
        reduction = config.get('reduction', 1)
        factor = config.get('factor')
        cardinality = config.get('cardinality')
        shortcut = config.get('shortcut', {})
        downsample = config.get('downsample', 1)
        filters, num_blocks = config.get('filters'), config.get('num_blocks')
        post_activation = config.get('post_activation', default=True)
        shape = input_shape
        body = Sequential()
        for i, ifilters in enumerate(filters):
            for j, repeats in enumerate(range(num_blocks[i])):
                x = self.block(input_shape=shape, filters=ifilters,
                               downsample=(j == 0 and i > 0) * downsample,
                               reduction=reduction, factor=factor,
                               cardinality=cardinality,
                               post_activation=post_activation,
                               use_se=use_se, use_bottleneck=use_bottleneck,
                               shortcut=shortcut)
                body.add_module("Block_{}_{}".format(i, j), x)
                shape = x.output_shape
        return body


class ResNet18(ResNet):

    @classmethod
    def default_config(cls):
        config = ResNet.default_config()
        body_config = {
            'factor': 1,
            'num_blocks': (2, 2, 2, 2),
            'use_bottleneck': False,
        }
        return config + {'body': body_config}


class ResNet34(ResNet):

    @classmethod
    def default_config(cls):
        config = ResNet.default_config()
        body_config = {
            'factor': 1,
            'num_blocks': (3, 4, 6, 3),
            'use_bottleneck': False,
        }
        return config + {'body': body_config}


class ResNet50(ResNet):

    @classmethod
    def default_config(cls):
        config = ResNet.default_config()
        body_config = {
            'num_blocks': (3, 4, 6, 3),
            'use_bottleneck': True,
        }
        return config + {'body': body_config}


class ResNext50(ResNet50):

    @classmethod
    def default_config(cls):
        config = ResNet50.default_config()
        config['body/cardinality'] = 32
        return config


class ResNet101(ResNet):

    @classmethod
    def default_config(cls):
        config = ResNet.default_config()
        body_config = {
            'num_blocks': (3, 4, 23, 3),
            'use_bottleneck': True,
        }
        return config + {'body': body_config}


class ResNext101(ResNet101):

    @classmethod
    def default_config(cls):
        config = ResNet101.default_config()
        config['body/cardinality'] = 32
        return config


class ResNet152(ResNet):

    @classmethod
    def default_config(cls):
        config = ResNet.default_config()
        body_config = {
            'num_blocks': (3, 8, 36, 3),
            'use_bottleneck': True,
        }
        return config + {'body': body_config}


class ResNext152(ResNet152):

    @classmethod
    def default_config(cls):
        config = ResNet152.default_config()
        config['body/cardinality'] = 32
        return config
