import torch
from torch.nn import functional as F
from ..layers.conv_block import ConvBlock
from ..layers.conv_block import ConvBranches
from ..bases import Sequential
from .base_model import BaseModel


class ResConvBranches(ConvBranches):

    def __init__(self, input_shape, scale=1.0, use_relu=True, **kwargs):
        super().__init__(input_shape=input_shape, mode='.', **kwargs)
        shape = super().output_shape
        self.scale = float(scale)
        self.use_relu = bool(use_relu)
        self.conv2d = ConvBlock(
            input_shape=shape, layout='c',
            c=dict(kernel_size=1, stride=1,
                   filters=self.in_channels)
        )

    @property
    def output_shape(self):
        return self.conv2d.output_shape

    def forward(self, x):
        y = super().forward(x)
        y = self.conv2d(y) * self.scale + x
        return F.relu(y) if self.use_relu else y


class InceptionResNetV2(BaseModel):

    @classmethod
    def mixed_5b(cls, input_shape):
        return ConvBranches(
            input_shape=input_shape, mode='.',
            branch0={
                'layout': 'cna',
                'c': {
                    'kernel_size': 1,
                    'stride': 1,
                    'filters': 96
                }
            },
            branch1={
                'layout': 'cna cna',
                'c': {
                    'kernel_size': [1, 5],
                    'stride': [1, 1],
                    'filters': [48, 64]
                }
            },
            branch2={
                'layout': 'cna cna cna',
                'c': {
                    'kernel_size': [1, 3, 3],
                    'stride': [1, 1, 1],
                    'filters': [64, 96, 96]
                }
            },
            branch3={
                'layout': 'p cna',
                'c': {
                    'kernel_size': 1,
                    'stride': 1,
                    'filters': 64
                },
                'p': {
                    'kernel_size': 3,
                    'stride': 1,
                    'mode': 'avg'
                }
            }
        )

    @classmethod
    def mixed_6a(cls, input_shape):
        return ConvBranches(
            input_shape=input_shape, mode='.',
            branch0={
                'layout': 'cna',
                'c': {
                    'kernel_size': 3,
                    'stride': 2,
                    'filters': 384
                }
            },
            branch1={
                'layout': 'cna cna cna',
                'c': {
                    'kernel_size': [1, 3, 3],
                    'stride': [1, 1, 2],
                    'filters': [256, 256, 384]
                }
            },
            branch2={
                'layout': 'p',
                'p': {
                    'kernel_size': 3,
                    'stride': 2,
                    'mode': 'max'
                }
            }
        )

    @classmethod
    def mixed_7a(cls, input_shape):
        return ConvBranches(
            input_shape=input_shape, mode='.',
            branch0={
                'layout': 'cna cna',
                'c': {
                    'kernel_size': [1, 3],
                    'stride': [1, 2],
                    'filters': [256, 384]
                }
            },
            branch1={
                'layout': 'cna cna',
                'c': {
                    'kernel_size': [1, 3],
                    'stride': [1, 2],
                    'filters': [256, 288]
                }
            },
            branch2={
                'layout': 'cna cna cna',
                'c': {
                    'kernel_size': [1, 3, 3],
                    'stride': [1, 1, 2],
                    'filters': [256, 288, 320]
                }
            },
            branch3={
                'layout': 'p',
                'p': {
                    'kernel_size': 3,
                    'stride': 2,
                    'mode': 'max'
                }
            }
        )

    @classmethod
    def block_35(cls, input_shape, scale=1.0, use_relu=True):
        return ResConvBranches(
            input_shape=input_shape,
            scale=scale, use_relu=use_relu,
            branch0={
                'layout': 'cna',
                'c': {
                    'kernel_size': 1,
                    'stride': 1,
                    'filters': 32
                }
            },
            branch1={
                'layout': 'cna cna',
                'c': {
                    'kernel_size': [1, 3],
                    'stride': [1, 1],
                    'filters': [32, 32]
                }
            },
            branch2={
                'layout': 'cna cna cna',
                'c': {
                    'kernel_size': [1, 3, 3],
                    'stride': [1, 1, 1],
                    'filters': [32, 48, 64]
                }
            }
        )

    @classmethod
    def block_17(cls, input_shape, scale=1.0, use_relu=True):
        return ResConvBranches(
            input_shape=input_shape,
            scale=scale, use_relu=use_relu,
            branch0={
                'layout': 'cna',
                'c': {
                    'kernel_size': 1,
                    'stride': 1,
                    'filters': 192
                }
            },
            branch1={
                'layout': 'cna cna cna',
                'c': {
                    'kernel_size': [(1, 1), (1, 7), (7, 1)],
                    'stride': [1, 1, 1],
                    'filters': [128, 160, 192]
                }
            }
        )

    @classmethod
    def block_8(cls, input_shape, scale=1.0, use_relu=True):
        return ResConvBranches(
            input_shape=input_shape,
            scale=scale, use_relu=use_relu,
            branch0={
                'layout': 'cna',
                'c': {
                    'kernel_size': 1,
                    'stride': 1,
                    'filters': 192
                }
            },
            branch1={
                'layout': 'cna cna cna',
                'c': {
                    'kernel_size': [(1, 1), (1, 3), (3, 1)],
                    'stride': [1, 1, 1],
                    'filters': [192, 224, 256]
                }
            }
        )

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()

        config['input'] = {
            'layout': 'cna cna cna p cna cna p',
            'c': dict(kernel_size=[3, 3, 3, 1, 3],
                      stride=[2, 1, 1, 1, 1],
                      filters=[32, 32, 64, 80, 192]),
            'p': dict(kernel_size=3, stride=2)
        }

        config['body'] = {
            'repeats': [10, 20, 10],
            'scales': [0.17, 0.10, 0.20]
        }

        config['head'] = {
            'layout': 'cna > fa',
            '>': dict(mode='avg'),
            'c': dict(kernel_size=1, stride=1, filters=1536),
            'f': dict(out_features=10),
            'a': dict(activation=['relu', 'linear'])
        }
        return config

    @classmethod
    def build_body(cls, input_shape, config):
        shape = input_shape
        repeats = config.get('repeats')
        scales = config.get('scales')

        assert len(repeats) == len(scales)

        body = Sequential()

        mixed_5b = cls.mixed_5b(shape)
        shape = mixed_5b.output_shape
        repeats_0 = Sequential(*[
            cls.block_35(shape, scale=scales[0])
            for i in range(repeats[0])
        ])

        mixed_6a = cls.mixed_6a(shape)
        shape = mixed_6a.output_shape
        repeats_1 = Sequential(*[
            cls.block_17(shape, scale=scales[1])
            for i in range(repeats[1])
        ])

        mixed_7a = cls.mixed_7a(shape)
        shape = mixed_7a.output_shape
        repeats_2 = Sequential(*([
            cls.block_8(shape, scale=scales[2])
            for i in range(repeats[2] - 1)]
            + [cls.block_8(shape, scale=scales[2], use_relu=False)])
        )

        body.add_module('mixed_5b', mixed_5b)
        body.add_module('repeats_0', repeats_0)
        body.add_module('mixed_6a', mixed_6a)
        body.add_module('repeats_1', repeats_1)
        body.add_module('mixed_7a', mixed_7a)
        body.add_module('repeats_2', repeats_2)
        return body
