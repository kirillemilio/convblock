""" Contains pytorch implementation of VGG architecture. """

from ..bases import Sequential
from .base_model import BaseModel


class VGG(BaseModel):

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()
        config['input_shape'] = (3, 224, 224)

        config['body'] = {
            'filters': (64, 128, 256, 512, 512),
            'num_blocks': (2, 2, 2, 2, 2),
            'kernel_size': (3, 3, 3, 3, 3),
            'pool_size': 3
        }

        config['head'] = {
            'num_classes': 10,
            'layout': 'p < fna fna fa',
            'f': dict(out_features=(4096, 4096, 1000)),
            'a': dict(activation=('relu', 'relu', 'linear')),
            'p': dict(kernel_size=2, stride=2, mode='max')
        }
        return config

    def build_body(self, input_shape, config):
        filters = config.get('filters')
        num_blocks = config.get('num_blocks')
        kernel_size = config.get('kernel_size')
        pool_size = config.get('pool_size')

        body = Sequential()
        shape = input_shape
        for i, ifilters in enumerate(filters):
            ilayout = 'cna' * num_blocks[i]
            ilayout = 'p' + ilayout if i > 0 else ilayout
            x = self.conv_block(
                input_shape=shape,
                layout=ilayout,
                c=dict(filters=ifilters, kernel_size=kernel_size[i]),
                p=dict(kernel_size=pool_size, stride=2, mode='max')
            )
            shape = x.output_shape
            body.add_module("Block_{}".format(i), x)
        return body


class VGG11(VGG):

    @classmethod
    def default_config(cls):
        config = VGG.default_config()
        body_config = {
            'num_blocks': (1, 1, 2, 2, 2)
        }
        return config + {'body': body_config}


class VGG13(VGG):

    @classmethod
    def default_config(cls):
        config = VGG.default_config()
        body_config = {
            'num_blocks': (2, 2, 2, 2, 2)
        }
        return config + {'body': body_config}


class VGG16D(VGG):

    @classmethod
    def default_config(cls):
        config = VGG.default_config()
        body_config = {
            'num_blocks': (2, 2, 3, 3, 3),
        }
        return config + {'body': body_config}


class VGG16C(VGG):

    @classmethod
    def default_config(cls):
        config = VGG.default_config()
        body_config = {
            'num_blocks': (2, 2, 3, 3, 3),
            'kernel_size': (3, 3, (3, 3, 1), (3, 3, 1), (3, 3, 1))
        }
        return config + {'body': body_config}


class VGG19(VGG):

    @classmethod
    def default_config(cls):
        config = VGG.default_config()
        body_config = {
            'num_blocks': (2, 2, 4, 4, 4)
        }
        return config + {'body': body_config}
