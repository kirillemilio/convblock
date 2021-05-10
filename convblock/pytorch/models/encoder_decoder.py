from collections import OrderedDict
import numpy as np
import torch

from ..bases import Sequential
from ..layers import ConvBlock

from .base_model import BaseModel
from ..blocks.nonlocal_block import NonLocalBlock
from ..blocks import BaseDecoder, VanillaUNetDecoder, VanillaVNetDecoder
from ..blocks import BaseEncoder, VanillaUNetEncoder, VanillaVNetEncoder


class EncoderDecoder(BaseModel):

    def __init__(self, config=None, *args, **kwargs):
        self.body_encoders = None
        self.body_decoders = None
        super().__init__(config, *args, **kwargs)

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()
        config['body_block'] = {
            'encoder': {
                'kernel_size': 3,
                'levels/filters': (64, 128, 256, 512, 1024),
                'levels/layout': ('cna', 'cna cna',
                                  'cna cna cna',
                                  'cna cna cna',
                                  'cna cna cna')
            },
            'decoder': {
                'kernel_size': 3,
                'levels/filters': (512, 256, 128, 64),
                'levels/layout': ('cna cna cna',
                                  'cna cna cna',
                                  'cna cna', 'cna')
            }
        }
        return config

    @property
    def encoders(self):
        return self.body_encoders

    @property
    def decoders(self):
        return self.body_decoders

    @classmethod
    def get_level(cls, config: 'Config', level: int) -> 'Config':
        return {key: value[level] for key, value in config.items()}

    @classmethod
    def get_levels_num(cls, config: 'Config') -> int:
        first_key = list(config.keys())[0]
        levels_num = len(config[first_key])
        if not np.all(np.array([len(v) for k, v in config.items()]) == levels_num):
            raise ValueError("Values in 'levels' part of config "
                             + "must have the same lenght")
        return levels_num

    @classmethod
    def encoder_block(cls, input_shape, **kwargs):
        x = torch.nn.Module()
        setattr(x, 'input_shape', input_shape)
        setattr(x, 'output_shape', input_shape)
        return x

    @classmethod
    def decoder_decoder(cls, input_shape, **kwargs):
        x = torch.nn.Module()
        setattr(x, 'input_shape', input_shape)
        setattr(x, 'output_shape', input_shape)
        return x

    def build_body(self, input_shape, config, **kwargs):
        encoder_config = {'block': self.conv_block}
        for key, value in config.get('encoder').items():
            if key != 'levels':
                encoder_config[key] = value

        decoder_config = {'block': self.conv_block}
        for key, value in config.get('decoder').items():
            if key != 'levels':
                decoder_config[key] = value

        encoders = Sequential()
        decoders = Sequential()

        encoders_output_shapes = []
        shape = input_shape
        for i in range(self.get_levels_num(config.get('encoder').get('levels'))):
            iconfig = {**encoder_config,
                       **self.get_level(config.get('encoder').get('levels'), i),
                       'input_shape': shape}
            iencoder = self.encoder_block(**iconfig)
            encoders.add_module('Encoder_{}'.format(i), iencoder)
            shape = iencoder.output_shape
            encoders_output_shapes.append(shape)

        for i in range(self.get_levels_num(config.get('decoder').get('levels'))):
            iconfig = {**decoder_config,  # noqa: E226
                       **self.get_level(config.get('decoder').get('levels'), i),
                       'input_shape': shape,
                       'skip_shape': encoders_output_shapes[-i-2]}
            idecoder = self.decoder_block(**iconfig)
            shape = idecoder.output_shape
            decoders.add_module('Decoder_{}'.format(i), idecoder)

        return encoders, decoders

    def build(self, *args, **kwargs):
        config = self.config
        input_shape = config.get('input_shape')
        self.input = self.build_input(input_shape=input_shape,
                                      config=config.get('input'))
        if self.input is not None:
            input_shape = self.input.output_shape

        self.body_encoders, self.body_decoders = self.build_body(
            input_shape=input_shape, config=config.get('body'))

        input_shape = self.body_decoders[-1].output_shape
        self.head = self.build_head(input_shape=input_shape,
                                    config=config.get('head'))
        if self.head is not None:
            input_shape = self.head.output_shape

        self.output_shape = input_shape

    def forward_decoders(self, inputs):
        x = inputs[-1]
        outputs = []
        for i, decoder in enumerate(self.decoders):
            x = decoder([x, inputs[-i-2]])
            outputs.append(x)
        return outputs

    def forward(self, inputs):
        x = self.input(inputs) if self.input else inputs
        *_, x = self.forward_decoders(self.forward_encoders(x))
        x = self.head(x) if self.head else x
        return x
