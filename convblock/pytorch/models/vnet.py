""" Contains pytorch implementation of VNet architecture. """

from ..blocks import VanillaVNetEncoder
from ..blocks import VanillaVNetDecoder
from .encoder_decoder import EncoderDecoder
from .base_model import BaseModel


class VNet(EncoderDecoder):

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()
        config['body'] = {
            'conv_block': {
                'a': dict(activation='prelu')
            },
            'encoder': {
                'kernel_size': 5,
                'levels': {
                    'filters': (64, 128, 256, 512, 1024),
                    'downsample': (False, True, True, True, True),
                    'layout': ('cna', 'cna cna', 'cna cna cna',
                               'cna cna cna', 'cna cna cna')
                }
            },
            'decoder': {
                'kernel_size': 5,
                'levels': {
                    'filters': (512, 256, 128, 64),
                    'layout': ('cna cna cna', 'cna cna cna',
                               'cna cna', 'cna')
                }
            }
        }
        config['head'] = {
            'layout': 'ca',
            'c': dict(filters=1, kernel_size=1, bias=True),
            'a': dict(activation='sigmoid')
        }
        return config

    @classmethod
    def encoder_block(cls, input_shape, filters, layout='cna', kernel_size=5,
                      downsample=False, downsampling_kernel=2,
                      post_activation=False, block=None):
        return VanillaVNetEncoder(
            input_shape, filters, layout, kernel_size,
            downsample, downsampling_kernel, post_activation, block)

    @classmethod
    def decoder_block(cls, input_shape, skip_shape, filters, layout='cna',
                      kernel_size=5, upsampling_kernel=2, upsampling_mode='t',
                      post_activation=False, block=None):
        return VanillaVNetDecoder(
            input_shape, skip_shape, filters, layout, kernel_size,
            upsampling_mode, upsampling_kernel, post_activation, block)
