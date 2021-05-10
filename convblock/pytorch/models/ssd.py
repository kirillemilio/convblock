import numpy as np
import torch
from torch.autograd import Variable
from ..layers import ConvBlock
from ..blocks.detector import DetectionEncoder
from ..blocks.detector import BaseLocClfPredictor
from .base_model import BaseModel
from .vgg import VGG
from .mobile_net_v1 import MobileNetV1
from .mobile_net_v2 import MobileNetV2
from ..config import Config
from ..bases import Module


class L2Norm(Module):
    def __init__(self, input_shape, scale=20):
        super(L2Norm, self).__init__(input_shape)
        self.n_channels = input_shape[0]
        self.gamma = scale
        self.eps = 1e-10
        self.weight = torch.nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    @property
    def output_shape(self):
        return self.input_shape

    def reset_parameters(self):
        torch.nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        x /= x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        out = (
            self.weight
            .unsqueeze(0)
            .unsqueeze(2)
            .unsqueeze(3)
            .expand_as(x) * x
        )
        return out


class SSDDetectionEncoder(DetectionEncoder):

    @classmethod
    def default_config(cls):
        return Config({
            'input_shape': (None, 512, 512),
            'num_levels': 6,
            'fmap_size': [38, 19, 10, 5, 3, 1],
            'tile_size': [30, 60, 111, 162, 213, 264, 315],
            'aspect_ratios': [(1.0, 0.5, 2.0),
                              (1.0, 0.333, 0.5, 2.0, 3.0),
                              (1.0, 0.333, 0.5, 2.0, 3.0),
                              (1.0, 0.333, 0.5, 2.0, 3.0),
                              (1.0, 0.5, 2.0),
                              (1.0, 0.5, 2.0)],
            'stride': [8, 16, 32, 64, 100, 300],
            'scales': [1.0],
            'variance': [0.1, 0.2],
            'normalize': True
        })

    @classmethod
    def generate_anchors(cls, fmap_size, tile_size, stride, ratios, scales):

        gx, gy, gr, gs = np.meshgrid(np.arange(fmap_size[0]),
                                     np.arange(fmap_size[1]),
                                     np.arange(len(ratios)),
                                     np.arange(len(scales)),
                                     indexing='ij')

        z = np.zeros(shape=(*fmap_size,
                            len(ratios),
                            len(scales), 4))

        z[gx, gy, gr, gs, 0] = ((gx + 0.5) * stride[0]
                                - 0.5 * tile_size[0, 0]
                                      * scales[gs]
                                      * np.sqrt(ratios[gr]))

        z[gx, gy, gr, gs, 1] = ((gy + 0.5) * stride[1]
                                - 0.5 * tile_size[1, 0]
                                      * scales[gs]
                                      * np.sqrt(1.0 / ratios[gr]))

        z[gx, gy, gr, gs, 2] = ((gx + 0.5) * stride[0]
                                + 0.5 * tile_size[0, 0]
                                      * scales[gs]
                                      * np.sqrt(ratios[gr]))

        z[gx, gy, gr, gs, 3] = ((gy + 0.5) * stride[1]
                                + 0.5 * tile_size[1, 0]
                                      * scales[gs]
                                      * np.sqrt(1.0 / ratios[gr]))

        b = np.zeros(shape=(fmap_size[0], fmap_size[1], len(scales), 4))
        bsize = np.sqrt([tile_size[0, 0] * tile_size[0, 1],
                         tile_size[1, 0] * tile_size[1, 1]])
        b[gx, gy, gs, 0] = (gx + 0.5) * stride[0] - 0.5 * bsize[0]
        b[gx, gy, gs, 1] = (gy + 0.5) * stride[1] - 0.5 * bsize[1]
        b[gx, gy, gs, 2] = (gx + 0.5) * stride[0] + 0.5 * bsize[0]
        b[gx, gy, gs, 3] = (gy + 0.5) * stride[1] + 0.5 * bsize[1]

        return np.concatenate([z, b[:, :, np.newaxis, :, :]], axis=2).reshape(-1, 4)

    def __init__(self, **config: dict):
        config = self.default_config() @ Config(config)

        input_shape = config['input_shape']

        num_levels = config.get('num_levels')
        aspect_ratios = np.array(config.get('aspect_ratios'))
        scales = np.array(config.get('scales'))

        if not isinstance(aspect_ratios[0], (tuple, list, np.ndarray)):
            aspect_ratios = np.tile(aspect_ratios[np.newaxis, :],
                                    [num_levels, 1])
        if not isinstance(scales[0], (tuple, list, np.ndarray)):
            scales = np.tile(scales[np.newaxis, :],
                             [num_levels, 1])

        stride = self._norm_param(config.get('stride', 8),
                                  int, num_levels, True)
        fmap_size = self._norm_param(config.get('fmap_size', 64),
                                     int, num_levels, True)

        tile_size = self._norm_param(config.get('tile_size', 0),
                                     int, num_levels + 1, True)

        if np.all(fmap_size == fmap_size[0, :]):
            fmap_size = np.ceil(fmap_size.T / 2 ** np.arange(num_levels)).T
            fmap_size = fmap_size.astype(np.int)

        if np.all(stride == stride[0, :]):
            stride = np.ceil(stride.T * 2 ** np.arange(num_levels)).T
            stride = stride.astype(np.int)

        if np.all(tile_size == tile_size[0, :]):
            tile_size = np.ceil(tile_size.T * 2 ** np.arange(num_levels + 1)).T
            tile_size = tile_size.astype(np.int)

        variance = config['variance']

        self.input_shape = input_shape
        self.fmap_size = fmap_size
        self.tile_size = tile_size
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.stride = stride

        priors = []
        tile_size = np.stack([tile_size[:-1, :],
                              tile_size[1:, :]], axis=2)
        for i in range(num_levels):
            x = self.generate_anchors(fmap_size[i, :],
                                      tile_size[i, :, :],
                                      stride[i, :],
                                      np.array(aspect_ratios[i]),
                                      np.array(scales[i]))
            priors.append(x)
        priors = np.concatenate(priors)
        if config.get('normalize', True):
            priors[:, [0, 2]] /= self.input_shape[1]
            priors[:, [1, 3]] /= self.input_shape[2]
        super(DetectionEncoder, self).__init__(Variable(torch.from_numpy(priors)),
                                               list(variance))


class SSD(BaseModel):

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
            'l2_norm': (True, False)
        }

        config['body'] = {
            'builder': ConvBlock,
            'params': {
                'layout': 'cna cna',
                'c': {
                    'kernel_size': [1, 3],
                    'stride': [1, 2],
                    'bias': False
                },
                'a': {
                    'activation': 'relu'
                },
                'levels': {
                    0: {'c/filters': [256, 512]},
                    1: {'c/filters': [128, 256]},
                    2: {'c/filters': [128, 256]},
                    3: {'c/filters': [128, 256]}
                }
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
        l2_norm = config.get('input/l2_norm', [])
        if len(l2_norm) < self.num_encoders:
            l2_norm = list(l2_norm) + [False] * \
                (self.num_encoders - len(l2_norm))
        self.l2_norm = l2_norm
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
        self.l2_norm = [L2Norm(model.encoders[idx].output_shape) if if_norm else None
                        for idx, if_norm in zip(self.encoders_indices, self.l2_norm)]
        return model

    def build_body(self, input_shape=None, config=None, **kwargs):
        builder = config.get('builder', self.conv_block)
        levels_config = config.pop('params/levels')
        last_encoder = self.input.encoders[self.encoders_indices[-1]]
        shape = last_encoder.output_shape
        body = torch.nn.ModuleList()
        for i in range(self.config['num_blocks']
                       - len(self.config['input/encoders'])):
            if levels_config is not None:
                iconfig = config['params'] @ levels_config[i]
            else:
                iconfig = config['params']
            x = builder(input_shape=shape, **iconfig)
            body.append(x)
            shape = x.output_shape
        return body

    def build_head(self, input_shape=None, config=None, **kwargs):
        builder = config.get('builder', self.conv_block)
        levels_config = config.pop('params/levels')
        encoders_shapes = [self.input.encoders[idx].output_shape
                           for idx in self.encoders_indices]
        fmap_shapes = [*encoders_shapes,
                       *[l.output_shape for l in self.body]]
        self.fmap_shapes = fmap_shapes
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
        if self.input is not None:
            input_shape = self.input.output_shape

        self.body = self.build_body(input_shape=input_shape,
                                    config=config.get('body'))

        self.head = self.build_head(input_shape=input_shape,
                                    config=config.get('head'))
        self.output_shape = None

    def forward(self, input_tensor):
        y = self.input.forward_encoders(input_tensor)
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


class SSD300(SSD):

    @classmethod
    def default_config(cls):
        config = SSD.default_config()
        config['input_shape'] = (3, 300, 300)
        config['input/input_shape'] = (3, 300, 300)
        return config


class SSD512(SSD):

    @classmethod
    def default_config(cls):
        config = SSD.default_config()
        config['input_shape'] = (3, 512, 512)
        config['input/input_shape'] = (3, 512, 512)
        config['num_blocks'] = 7

        config['encoder'] = {
            'num_levels': 7,
            'fmap_size': [64, 32, 16, 8, 4, 2, 1],
            'tile_size': [20.48, 61.2, 133.12, 215.04,
                          296.96, 378.88, 460.8, 542.72],
            'aspect_ratios': [[1.0 / 2.0, 1.0, 2.0],
                              [1.0 / 3.0, 1.0 / 2.0, 1.0, 2.0, 3.0],
                              [1.0 / 3.0, 1.0 / 2.0, 1.0, 2.0, 3.0],
                              [1.0 / 3.0, 1.0 / 2.0, 1.0, 2.0, 3.0],
                              [1.0 / 3.0, 1.0 / 2.0, 1.0, 2.0, 3.0],
                              [1.0 / 2.0, 1.0, 2.0],
                              [1.0 / 2.0, 1.0, 2.0]],
            'stride': [8, 16, 32, 64, 128, 256, 512],
            'scales': [1.0],
            'variance': [0.1, 0.2]
        }

        config['body'] = {
            'params': {
                'layout': 'cna cna',
                'c': {
                    'kernel_size': [1, 3],
                    'stride': [1, 2],
                    'bias': False
                },
                'a': {
                    'activation': 'relu'
                },
                'levels': {
                    0: {'c/filters': [512, 1024]},
                    1: {'c/filters': [256, 512]},
                    2: {'c/filters': [128, 256]},
                    3: {'c/filters': [128, 256]},
                    4: {'c/filters': [128, 256]}
                }
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


class SSDLiteV1(SSD300):

    @classmethod
    def default_config(cls):
        config = SSD300.default_config()

        del config['input']
        del config['body']
        del config['head']

        config['encoder'] = {
            'num_levels': 6,
            'fmap_size': [19, 10, 5, 3, 2, 1],
            'tile_size': [45, 90, 135, 180, 225, 270, 315],
            'aspect_ratios': [[1.0 / 3.0, 1.0 / 2.0, 1.0, 2.0, 3.0],
                              [1.0 / 3.0, 1.0 / 2.0, 1.0, 2.0, 3.0],
                              [1.0 / 3.0, 1.0 / 2.0, 1.0, 2.0, 3.0],
                              [1.0 / 3.0, 1.0 / 2.0, 1.0, 2.0, 3.0],
                              [1.0 / 2.0, 1.0, 2.0],
                              [1.0 / 2.0, 1.0, 2.0]],
            'stride': [16, 32, 64, 100, 150, 300],
            'scales': [1.0],
            'variance': [0.1, 0.2]
        }

        config['input'] = {
            'model': MobileNetV1,
            'config': MobileNetV1.default_config() @ {'head': None},
            'encoders': (10, 12),
            'l2_norm': (False, False)
        }

        config['body'] = {
            'builder': MobileNetV1.block,
            'params': {
                'downsample': True,
                'layout': 'cna',
                'levels': {
                    0: {'filters': 512},
                    1: {'filters': 256},
                    2: {'filters': 256},
                    3: {'filters': 128},
                }
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


class SSDLiteV2(SSD300):

    @classmethod
    def default_config(cls):
        config = SSD300.default_config()

        del config['input']
        del config['body']
        del config['head']

        config['encoder'] = {
            'num_levels': 6,
            'fmap_size': [19, 10, 5, 3, 2, 1],
            'tile_size': [45, 90, 135, 180, 225, 270, 315],
            'aspect_ratios': [[1.0 / 3.0, 1.0 / 2.0, 1.0, 2.0, 3.0],
                              [1.0 / 3.0, 1.0 / 2.0, 1.0, 2.0, 3.0],
                              [1.0 / 3.0, 1.0 / 2.0, 1.0, 2.0, 3.0],
                              [1.0 / 3.0, 1.0 / 2.0, 1.0, 2.0, 3.0],
                              [1.0 / 2.0, 1.0, 2.0],
                              [1.0 / 2.0, 1.0, 2.0]],
            'stride': [16, 32, 64, 100, 150, 300],
            'scales': [1.0],
            'variance': [0.1, 0.2],
        }

        config['input'] = {
            'model': MobileNetV2,
            'config': MobileNetV2.default_config() @ {'head': None},
            'encoders': (12, 16),
            'l2_norm': (False, False)
        }

        config['body'] = {
            'builder': MobileNetV2.block,
            'params': {
                'downsample': True,
                'layout': 'cna cna cn',
                'levels': {
                    0: {'filters': 512},
                    1: {'filters': 256},
                    2: {'filters': 256},
                    3: {'filters': 128},
                }
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


class MobileSSD300V1(SSD300):

    @classmethod
    def default_config(cls):
        config = SSD300.default_config()

        del config['input']
        del config['body']
        del config['head']

        config['input'] = {
            'model': MobileNetV1,
            'config': MobileNetV1.default_config() @ {'head': None},
            'encoders': (4, 10, 12),
            'l2_norm': (True, False)
        }

        config['body'] = {
            'builder': MobileNetV1.block,
            'params': {
                'downsample': True,
                'layout': 'cna',
                'levels': {
                    0: {'filters': 512},
                    1: {'filters': 256},
                    2: {'filters': 256},
                }
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


class MobileSSD512V1(SSD512):

    @classmethod
    def default_config(cls):
        config = SSD512.default_config()

        del config['input']
        del config['body']
        del config['head']

        config['input'] = {
            'model': MobileNetV1,
            'config': MobileNetV1.default_config() @ {'head': None},
            'encoders': (4, 10, 12),
            'l2_norm': (True, False, False)
        }

        config['body'] = {
            'builder': MobileNetV1.block,
            'params': {
                'downsample': True,
                'layout': 'cna',
                'levels': {
                    0: {'filters': 512},
                    1: {'filters': 256},
                    2: {'filters': 256},
                    3: {'filters': 256}
                }
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


class MobileSSD300V2(SSD300):

    @classmethod
    def default_config(cls):
        config = SSD300.default_config()

        del config['input']
        del config['body']
        del config['head']

        config['input'] = {
            'model': MobileNetV2,
            'config': MobileNetV2.default_config() @ {'head': None},
            'encoders': (5, 12, 16),
            'l2_norm': (True, False, False)
        }

        config['body'] = {
            'builder': MobileNetV2.block,
            'params': {
                'downsample': True,
                'layout': 'cna cna cn',
                'levels': {
                    0: {'filters': 512},
                    1: {'filters': 256},
                    2: {'filters': 256},
                }
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


class MobileSSD512V2(SSD512):

    @classmethod
    def default_config(cls):
        config = SSD512.default_config()

        del config['input']
        del config['body']
        del config['head']

        config['input'] = {
            'model': MobileNetV2,
            'config': MobileNetV2.default_config() @ {'head': None},
            'encoders': (5, 12, 16),
            'l2_norm': (True, False, False)
        }

        config['body'] = {
            'builder': MobileNetV2.block,
            'params': {
                'downsample': True,
                'layout': 'cna cna cn',
                'levels': {
                    0: {'filters': 512},
                    1: {'filters': 256},
                    2: {'filters': 256},
                    3: {'filters': 256}
                }
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
