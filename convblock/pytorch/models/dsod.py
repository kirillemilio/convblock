# "DSOD: Learning Deeply Supervised Object Detectors from Scratch" Zhiqiang Shen et al. ICCV17

import math
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import ConvBlock
from ..bases import Module
from ..bases import Sequential
from ..blocks import DenseBlock
from ..blocks import BaseResBlock
from ..blocks.detector import SimpleLocClfPredictor
from ..blocks.detector import DetectionEncoder
from .ssd import L2Norm
from .base_model import BaseModel

from ..blocks import DenseBlock
from ..bases import Sequential
from .base_model import BaseModel
from ..bases import Branches, Stack, Cat


class DSODBackbone(Module):

    @classmethod
    def transition(cls, input_shape,
                   filters: int,
                   layout: str = 'cna',
                   downsample: bool = True,
                   block: 'ConvBlock' = None,
                   **config):
        conv_block = ConvBlock if block is None else block
        if downsample:
            layout = layout + '/p'
        return conv_block(
            input_shape=input_shape,
            layout=layout,
            **config
        )

    @classmethod
    def supervision(cls, input_shape,
                    filters: int,
                    left_layout: str = 'p nac',
                    right_layout: str = 'nac nac',
                    pool_size: 'ArrayLike[int]' = 2,
                    block: 'ConvBlock' = None):
        conv_block = ConvBlock if block is None else block
        return Branches([
            conv_block(input_shape=input_shape,
                       layout=left_layout,
                       c=dict(kernel_size=1,
                              fitlers=filters,
                              stride=1)),
            conv_block(
                input_shape=input_shape,
                layout=right_layout,
                c=dict(kernel_size=[1, 3],
                       fitlers=filters,
                       stride=[1, 2]),
                p=dict(kernel_size=pool_size,
                       stride=2, mode='max')
            )
        ])

    def __init__(self, input_shape: 'ArrayLike[int]',
                 num_layers: 'ArrayLike[int]',
                 growth_rate: int = 48,
                 filters: tuple = (256, 256, 128, 128, 128),
                 bottleneck_factor: int = 4,
                 compression_factor: float = 1.0,
                 block: 'ConvBlock' = None):
        super().__init__(input_shape)
        conv_block = ConvBlock if block is None else block
        dense_block = DenseBlock.p(block=conv_block,
                                   use_bottleneck=True,
                                   growth_rate=growth_rate,
                                   kernel_size=kernel_size)
        self.dense1 = dense_block(
            input_shape=self.input_shape,
            num_layers=num_layers[0],
        )
        self.trans1 = self.transition(
            input_shape=self.dense1.output_shape,
            filters=math.ceil(growth_rate
                              * compression_factor),
            downsample=True
        )
        self.dense2 = dense_block(
            input_shape=self.trans1.output_shape[1],
            num_layers=num_layers[1]
        )
        self.trans2 = self.transition(
            input_shape=self.dense2.output_shape,
            filters=math.ceil(growth_rate
                              * compression_factor),
            downsample=True
        )
        self.dense3 = dense_block(
            input_shape=self.trans2.output_shape[1],
            num_layers=num_layers[2],
        )
        self.trans3 = self.transition(
            input_shape=self.dense3.output_shape,
            filters=math.ceil(growth_rate
                              * compression_factor),
            downsample=False
        )
        self.dense4 = dense_block(
            input_shape=self.trans3.output_shape,
            num_layers=num_layers[3]
        )
        self.trans4 = self.transition(
            input_shape=self.dense4.output_shape,
            filters=math.ceil(growth_rate
                              * compression_factor),
            downsample=False
        )
        self.super1 = conv_block(
            input_shape=self.trans2.output_shape,
            layout='p nac',
            p=dict(kernel_size=pool_size, stride=2),
            c=dict(kernel_size=1, stride=1,
                   filters=filters[0])
        )
        self.super2 = self.supervision(
            input_shape=shape,
            filters=filters[1]
        )
        self.super3 = self.supervision(
            input_shape=self.super2.output_shape,
            filters=filters[2]
        )
        self.super4 = self.supervision(
            input_shape=self.super3.output_shape,
            filters=filters[3]
        )

    def forward(self, x):
        x = self.dense1(x)
        _, x = self.trans1(x)

        x = self.dense2(x)
        f1, x = self.trans2(x)

        x = self.dense3(x)
        x = self.trans3(x)

        x = self.dense4(x)
        x = self.trans4(x)

        f2 = torch.cat([self.super1(f1), x], 1)
        f3 = self.super2(f2)
        f4 = self.super3(f3)
        f5 = self.super4(f4)
        f6 = self.super5(f5)

        return f1, f2, f3, f4, f5, f6


class DSODBackbone(Module):

    @classmethod
    def lhrh(cls, input_shape,
             filters: int,
             layout: str = 'nac',
             factor: float = 1.0,
             pool_size: int = 2):
        body = ConvBlock.partial(
            layout=layout * 2,
            c=dict(filters=(filters // 2 * factor,
                            filters // 2), kernel_size=[1, 3]),

        )
        shortcut = ConvBlock.partial(
            layout='p ' + layout,
            p=dict(kernel_size=pool_size, stride=1),
            c=dict(filters=filters // 2, kernel_size=1)
        )
        return BaseResBlock(input_shape=input_shape,
                            body=body, shortcut=shortcut, how='.')

    @classmethod
    def default_config(cls):
        return Config({
            'input': {
                'layout': 'cna cna cna p',
                'c': {
                    'filters': (64, 64, 128),
                    'kernel_size': 3,
                    'stride': (2, 1, 1),
                },
                'p': {
                    'kernel_size': 2,
                    'stride': 1
                }
            },
            'dense': {
                'layout': 'nac',
                'kernel_size': 3,
                'bottleneck_factor': 4,
                'use_bottleneck': True,
                'growth_rate': 48,
                'num_layers': 6
            },
            'transition': {
                'layout': 'cna p',
                'c': {
                    'kernel_size': 3
                },
                'p': {
                    'kernel_size': 1,
                    'mode': 'max'
                }
            },
            'lhrh': {
                'layout': 'nac',
                'factor': 1.0,
                'pool_size': 2
            }
        })

    def __init__(self, input_shape,
                 depth=(6, 8, 8, 8),
                 extra_filters=(512, 256, 256, 256),
                 out_filters: int = 256,
                 growth_rate=48, config: dict = None):
        super(DSOD, self).__init__(input_shape)
        config = self.default_config() @ Config(config) @ {
            'dense/growth_rate': growth_rate,
        }
        filters = [128 + growth_rate * _
                   for _ in np.cumsum(depth)]

        out_filters = int(out_filters)

        self.input = ConvBlock(
            input_shape=self.input_shape,
            **config['input']
        )

        self.d11 = DenseBlock(
            input_shape=self.input.output_shape,
            **(config['dense'] @ {'num_layers': depth[0]})
        )
        self.t11 = ConvBlock(
            input_shape=self.d11.output_shape,
            **(config['transition']
               @ {'c/filters': filters[0], 'p/stride': 2})
        )
        self.d12 = DenseBlock(
            input_shape=self.t11.output_shape,
            **(config['dense'] @ {'num_layers': depth[1]})
        )
        self.t12 = ConvBlock(
            input_shape=self.d12.output_shape,
            **(config['transition']
               @ {'c/filters': filters[1]})
        )

        self.pool2 = ConvBlock(
            input_shape=self.t12.output_shape,
            layout='p', p=dict(kernel_size=2,
                               stride=1)
        )
        self.conv2 = ConvBlock(
            input_shape=self.pool2.output_shape,
            layout='nac', c=dict(filters=out_filters,
                                 kernel_size=3,
                                 stride=1)
        )

        self.d21 = DenseBlock(
            input_shape=self.pool2.output_shape,
            **(config['dense'] @ {'num_layers': depth[2]})
        )
        self.t21 = ConvBlock(
            input_shape=self.d21.output_shape,
            **(config['transition']
               @ {'c/filters': filters[2]})
        )
        self.d22 = DenseBlock(
            input_shape=self.t21.output_shape,
            **(config['dense'] @ {'num_layers': depth[3]})
        )
        self.t22 = ConvBlock(
            input_shape=self.d21.output_shape,
            **(config['transition']
               @ {'c/filters': out_filters})
        )

        self.extra = torch.nn.ModuleList()
        shape = np.array([self.conv2.output_shape[0]
                          + self.t22.output_shape[0],
                          *self.conv2.output_shape[1:]])
        for ifilters in extra_filters:
            x = self.lhrh(input_shape=shape,
                          filters=ifilters)
            self.extra.append(x)

        self.output_shape = [self.t12.output_shape, shape,
                             *[m.output_shape
                               for m in self.extra]]
        self.l2norm = torch.nn.ModuleList([L2Norm(shape)
                                           for shape in self.output_shape])

    def forward(self, x):
        outputs = []

        x = self.input(x)
        x = self.t11(self.d11(x))
        x = self.t12(self.d12(x))

        outputs.append(x)

        z = self.pool2(x)
        y = self.conv2(z)
        z = self.t21(self.d21(z))
        z = self.t22(self.d22(z))
        x = torch.cat([y, z], 1)
        outputs.append(x)

        for m in self.extra:
            x = m(x)
            outputs.append(x)
        return [norm(x) for norm, x in zip(self.l2norm, outputs)]


class DSOD(BaseModel):

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()

        config['num_classes'] = 20
        config['num_blocks'] = 6

        config['input_shape'] = (3, 300, 300)

        config['encoder'] = {
            'num_levels': 6,
            'fmap_size': [38, 19, 10, 5, 3, 2],
            'tile_size': [30, 60, 111, 162, 213, 264, 315],
            'aspect_ratios': [[1.0 / 2.0, 1.0, 2.0],
                              [1.0 / 3.0, 1.0 / 2.0, 1.0, 2.0, 3.0],
                              [1.0 / 3.0, 1.0 / 2.0, 1.0, 2.0, 3.0],
                              [1.0 / 3.0, 1.0 / 2.0, 1.0, 2.0, 3.0],
                              [1.0 / 2.0, 1.0, 2.0],
                              [1.0 / 2.0, 1.0, 2.0]],
            'stride': [8, 16, 32, 64, 100, 300],
            'scales': [1.0],
            'variance': [0.1, 0.2]
        }

        config['input'] = None

        config['body'] = {
            'depth': (6, 8, 8, 8),
            'extra_filters': (512, 256, 256, 256),
            'out_filters': 256,
            'growth_rate': 48
        }

        config['head'] = {
            'builder': BaseLocClfPredictor,
            'params': BaseLocClfPredictor.default_config() @ {
                'clf/layout': 'c',
                'loc/layout': 'c'
            }
        }
        return config

    def __init__(self, config: dict = None):
        config = self.default_config() @ Config(config)
        self.num_classes = config.get('num_classes')
        self.num_anchors = (len(config['encoder/scales'])
                            * len(config['encoder/aspect_ratios']))
        super().__init__(config)

    def encode(self,
               bboxes: 'Tensor',
               classes: 'Tensor',
               threshold: float = 0.5):
        return self.encoder.encode(bboxes, classes, threshold)

    def decode(self,
               loc: 'Tensor',
               conf: 'Tensor',
               threshold: float = 0.5):
        return self.encoder.decode(loc, conf, threshold)

    def build_input(self, input_shape=None, config=None):
        self.encoder = DetectionEncoder(**(self.config['encoder']
                                           @ {'input_shape': input_shape}))
        return None

    def build_body(self, input_shape=None, config=None):
        return DSODBackbone(input_shape=self.config['input_shape'], **config)

    def build_head(self, input_shape=None, config=None):
        shapes = self.body.output_shape[...]
        return torch.nn.ModuleList([
            SimpleLocClfPredictor(shapes[i, :],
                                  self.num_classes,
                                  self.num_anchors,
                                  config)
            for i in range(shapes.shape[0])
        ])

    def build(self, *args, **kwargs):
        config = self.config
        input_shape = config.get('input_shape')
        self.input = self.build_input(input_shape=input_shape,
                                      config=config.get('input'))

        self.body = self.build_body(input_shape=input_shape,
                                    config=config.get('body'))

        self.head = self.build_head(input_shape=None,
                                    config=config.get('head'))

        self.output_shape = None

    def forward(self, x):
        outputs = [head(x) for head, x in zip(self.head, self.body(x))]
        return (torch.cat([x[0] for x in outputs], 1),
                torch.cat([x[1] for x in outputs], 1))
