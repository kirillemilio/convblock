import math
import numpy as np
import torch
from ..config import Config
from .base_model import BaseModel
from .fpn import FeaturesPyramid
from .resnet import ResNet50
from ..blocks.detector import SimpleLocClfPredictor
from ..blocks.detector import DetectionEncoder


class RetinaNet(BaseModel):

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()

        config['num_classes'] = 20
        config['input_shape'] = (3, 512, 512)

        config['encoder'] = {
            'num_levels': 5,
            'num_levels': 5,
            'tile_size': [32, 64, 128, 256, 512],
            'fmap_size': [64, 32, 16, 8, 4],
            'stride': [8, 16, 32, 64, 128],
            'scales': [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
            'aspect_ratios': [1.0 / 2.0, 1.0, 2.0],
            'variance': [0.1, 0.2]
        }

        config['input'] = {
            'model': ResNet50,
            'config': {
                'head': None
            },
            'encoders': (6, 12, 15)
        }

        config['body'] = {
            'in_filters': 256,
            'out_filters': 256,
            'extra_filters': 256,
            'num_extra': 2,
            'config': FeaturesPyramid.default_config()
        }

        config['head'] = {
            'body': {
                'layout': 'cna cna cna cna',
                'c': {
                    'kernel_size': 3,
                    'bias': False,
                    'filters': [256, 256, 256, 256]
                },
                'a': {
                    'activation': 'relu'
                }
            },
            'clf/c': {
                'init_weight': lambda w: w.data.fill_(-math.log((1.0 - 0.1) / 0.1)),
                'init_bias': lambda b: b.data.fill_(0)
            }
        }
        return config

    def __init__(self, config: dict = None):
        config = self.default_config() @ Config(config)
        encoders = config.get('input/encoders')
        self.encoders_indices = np.array(encoders)
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

    def build_input(self, input_shape, config=None):
        self.encoder = DetectionEncoder(**(self.config['encoder']
                                           @ {'input_shape': input_shape}))
        input_config = config['config'] @ {'input_shape': input_shape}
        return config['model'](input_config)

    def build_body(self, input_shape=None, config=None):
        fpn_input_shape = np.stack([self.input.encoders[i].output_shape
                                    for i in self.encoders_indices], axis=0)

        return FeaturesPyramid(fpn_input_shape, **config)

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
        if self.input is not None:
            input_shape = self.input.output_shape

        self.body = self.build_body(input_shape=None,
                                    config=config.get('body'))

        self.head = self.build_head(input_shape=None,
                                    config=config.get('head'))

    def forward(self, x):
        model_outputs = np.array(self.input.forward_encoders(x))
        fpn_outputs = self.body(model_outputs[self.encoders_indices].tolist())
        outputs = []
        for predictor, output in zip(self.head, fpn_outputs):
            outputs.append(predictor(output))
        loc, conf = zip(*outputs)
        return torch.cat(loc, 1), torch.cat(conf, 1)
