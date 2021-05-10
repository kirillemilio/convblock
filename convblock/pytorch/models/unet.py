""" Contains pytorch implementation of UNet architecture. """

from ..bases import Sequential
from ..layers import ConvBlock
from ..blocks import VanillaUNetEncoder
from ..blocks import VanillaUNetDecoder
from ..blocks import GCNBlock
from ..blocks import NonLocalBlock
from ..blocks import BaseDecoder
from .encoder_decoder import EncoderDecoder
from .base_model import BaseModel


class UNet(EncoderDecoder):

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()
        config['body'] = {
            'encoder': {
                'kernel_size': 3,
                'levels': {
                    'filters': (64, 128, 256, 512, 1024),
                    'downsample': (False, True, True, True, True),
                    'layout': ('cna cna', ) * 5
                }
            },
            'decoder': {
                'kernel_size': 3,
                'levels': {
                    'filters': (512, 256, 128, 64),
                    'layout': ('cna cna', ) * 4
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
    def encoder_block(cls, input_shape, filters, layout='cna', kernel_size=3,
                      downsample=False, downsampling_kernel=2, block=None):
        return VanillaUNetEncoder(
            input_shape, filters, layout, kernel_size,
            downsample, downsampling_kernel, block)

    @classmethod
    def decoder_block(cls, input_shape, skip_shape, filters,
                      layout='cna cna', kernel_size=3, upsampling_mode='t',
                      upsampling_kernel=2, block=None):
        return VanillaUNetDecoder(
            input_shape, skip_shape, filters, layout, kernel_size,
            upsampling_mode, upsampling_kernel, block)


class UNetGCN(UNet):

    @classmethod
    def default_config(cls):
        config = UNet.default_config()
        config['body/encoder/kernel_size'] = 11
        return config

    @classmethod
    def encoder_block(cls, input_shape, filters, kernel_size=11, downsample=False):
        shape = input_shape
        encoder_layers = Sequential()
        if downsample:
            x = ConvBlock(
                input_shape=shape, layout='p',
                p=dict(kernel_size=2, stride=2)
            )
            encoder_layers.add_module('Downsample', x)
            shape = x.output_shape

        x = GCNBlock(
            input_shape=shape, layout='cc',
            kernel_size=kernel_size, filters=filters // 2, how='.'
        )

        encoder_layers.add_module('GCBody', x)
        encoder_layers.add_module('GCNHead', ConvBlock(input_shape=x.output_shape,
                                                       layout='na'))
        return encoder_layers

    @classmethod
    def decoder_block(cls, input_shape, skip_shape, filters, kernel_size=11):
        input_block = ConvBlock.partial(
            layout='tna', t=dict(kernel_size=2, stride=2, filters=filters)
        )

        body = GCNBlock.partial(
            filters=filters // 2, layout='cc',
            kernel_size=kernel_size, how='.'
        )

        decoder = BaseDecoder(
            input_shape=input_shape, skip_shape=skip_shape,
            input_block=input_block, body=body, how='.'
        )

        return Sequential(decoder, ConvBlock(input_shape=decoder.output_shape,
                                             layout='na'))


class NonLocalEncoderUNet(UNet):
    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()
        config['body'] = {
            'encoder': {
                'levels': {
                    'filters': (64, 128, 256, 512, 1024),
                    'downsample': (False, True, True, True, True),
                    'non_local': (False, False, True, True, True)
                }
            },
            'decoder': {
                'levels': {
                    'filters': (512, 256, 128, 64),
                    'non_local': (True, True, True, False)
                }
            }
        }
        return config

    @classmethod
    def encoder_block(cls, input_shape, filters, kernel_size=3,
                      downsample=False, non_local=False):
        encoder_layers = Sequential()

        x = ConvBlock(
            input_shape=input_shape, layout='p cna' if downsample else 'cna',
            c=dict(kernel_size=kernel_size, filters=filters),
            p=dict(kernel_size=2, stride=2)
        )
        encoder_layers.add_module('InputConv', x)

        if non_local:
            encoder_layers.add_module(
                'NonLocal', NonLocalBlock(
                    input_shape=x.output_shape, filters=[filters, filters])
            )
        else:
            encoder_layers.add_module(
                'Conv', ConvBlock(
                    input_shape=x.output_shape, layout='cna',
                    c=dict(filters=filters, kernel_size=kernel_size))
            )
        return encoder_layers
