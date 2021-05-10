import numpy as np
import torch
from torch.autograd import Variable
from ..layers import ConvBlock
from ..blocks import RFB, RFBa
from .base_model import BaseModel
from .vgg import VGG
from ..config import Config
from ..bases import Module


class RFBPyramid(Pyramid):

    def __init__(self, input_shape, input_layout='sa',
                 extra_layout='sscc', num_extra=2):
        self.input_shape = np.array(input_shape)
        num_inputs = len(self.input_shape)

        if num_extra == 0:
            extra_filters = []
        else:
            extra_filters = transform_to_int_tuple(extra_filters,
                                                   'extra_filters',
                                                   num_extra)
        config = self.default_config() @ Config(config)


class RFBSSD(BaseModel):

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()

        config['num_classes'] = 20
        config['num_blocks'] = 6

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

        config['input'] = {
            'model': VGG,
            'config': {
                'body': {
                    'filters': (64, 128, 256, 512, 512),
                    'num_blocks': (2, 2, 3, 3, 3),
                    'kernel_size': (3, 3, 3, 3, 3),
                    'pool_size': 2
                },
                'head': None
            },
            'encoders': (3, 4),
        }

        config['body'] = {
            'levels': {
                0: {'builder': RFB, 'params': {'downsample': True}},
                1: {'builder': RFB, 'params': {'downsample': True}},
                2: {'builder': ConvBlock, 'params': {'layout': 'cna cna',
                                                     'kernel_size': [1, 3],
                                                     'stride': [1, 2]}},
                3: {'builder': ConvBlock, 'params': {'layout': 'cna cna',
                                                     'kernel_size': [1, 3],
                                                     'stride': [1, 2]}}
            }
        }

        config['head'] = {
            'builder': BaseLocClfPredictor,
            'params': BaseLocClfPredictor.default_config() @ {
                'clf/layout': 'c',
                'loc/layout': 'c'
            }
        }
        return config

    def __init__(self, config: dict = None, *args, **kwargs):
        config = self.default_config() @ Config(config)
        self.encoders_indices = np.array(config.get('input/encoders'),
                                         dtype=np.int)
        self.encoders_indices = np.sort(self.encoders_indices)
        self.num_encoders = len(self.encoders_indices)
        self.num_classes = config.get('num_classes')
        if 'input_shape' in config:
            config = Config(config) @ {
                'input/config/input_shape': config['input_shape']
            }
        super().__init__(config, *args, **kwargs)

    def build_input(self, input_shape, config=None, **kwargs):
        self.encoder = SSDDetectionEncoder(**(self.config['encoder']
                                              @ {'input_shape': input_shape}))
        base, params = config.get('model'), config.get('config')
        model = base({'input_shape': input_shape, **params})
        indices = config['encoders']

        encoders = []
        if model.input is not None:
            encoders += [Sequential(model.input, *model.body[:indices[0]])]
        else:
            encoders += [Sequential(model.body[:indices[0] + 1])]

        for prev_idx, next_idx in zip(indices[:-1], indices[1:]):
            encoders += [Sequential(*model.body[prev_idx + 1, next_idx + 1])]

        encoders[-1] = Sequential(*encoders[-1],
                                  RBF(encoders[-1].output_shape))
        encoders[-2] = Sequential(*encoders[-2],
                                  RBFs(encoders[-2].output_shape))

        return torch.nn.ModuleList(encoders)

    def build_body(self, input_shape=None, config=None, **kwargs):
        shape = self.input[-1].output_shape
        body = torch.nn.ModuleList()
        for i in range(len(config['levels'])):
            iconfig = config.get('levels').get(i)
            ibuilder = iconfig['builder']
            iblock = ibuilder(**(iconfig['params']
                                 @ {'input_shape': shape}))
            shape = iblock.output_shape
            body.append(iblock)
        return body

    def build_head(self, input_shape=None, config=None, **kwargs):
        builder = config.get('builder', self.conv_block)
        levels_config = config.pop('params/levels')
        shapes = [x.output_shape for x in self.input]
        shapes += [x.output_shape for x in self.body]
        self.fmap_shapes = shapes
        aspect_ratios = self.config['encoder/aspect_ratios']

        head = torch.nn.ModuleList()
        for i in range(self.config['num_blocks']):
            if levels_config is not None:
                iconfig = config['params'] @ levels_config[i]
            else:
                iconfig = config['params']
            x = builder(input_shape=fmap_shapes[i],
                        num_classes=self.num_classes,
                        num_anchors=len(aspect_ratios[i]) + 1,
                        config=iconfig)
            head.append(x)
        return head

    def build(self, *args, **kwargs):
        config = self.config
        input_shape = config.get('input_shape')
        self.input = self.build_input(input_shape=input_shape,
                                      config=config.get('input'))
        self.body = self.build_body(config=config.get('body'))
        self.head = self.build_head(config=config.get('head'))
        self.output_shape = None

    def forward(self, input_tensor):
        outputs = [module(x)
                   for module, x in zip(self.input, self.input_tensor)]
        outputs = [(self.head[i](y[idx])
                    if self.l2_norm[i] is None
                    else self.head[i](self.l2_norm[i](y[idx])))
                   for i, idx in enumerate(self.encoders_indices)]
        x = y[-1]
        for i, (extra_, head_) in enumerate(zip(self.body,
                                                self.head[self.num_encoders:])):
            x = extra_(x)
            outputs.append(head_(x))
        return (torch.cat([x[0] for x in outputs], 1),
                torch.cat([x[1] for x in outputs], 1))
