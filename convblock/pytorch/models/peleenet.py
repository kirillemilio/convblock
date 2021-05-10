""" Contains implementation of PeleeNet architecture. """

from ..blocks import PeleeDenseBlock
from ..bases import Sequential
from .base_model import BaseModel


class PeleeNet(BaseModel):

    @classmethod
    def default_config(cls):
        """ Get default config for PeleeNet model. """
        config = BaseModel.default_config()
        config['input'] = {
            'layout': 'cna .cna cna. cna',
            'c': dict(kernel_size=[3, 1, 3, 1],
                      stride=[2, 1, 2, 1],
                      filters=[32, 16, 32, 32]),
            'shortcut': dict(layout='p', pool_size=2)
        }

        config['body/dense'] = {
            'num_layers': (3, 4, 8, 6),
            'growth_rate': (32, 32, 32, 32),
            'bottleneck_factor': (1, 2, 4, 4),
            'layout': 'cna',
            'kernel_size': 3,
        }

        config['body/transition'] = {
            'layout': 'cnap',
            'c': dict(kernel_size=1, stride=1),
            'p': dict(kernel_size=2, stride=2, mode='avg')
        }

        config['head'] = {
            'layout': '> fa',
            'f': dict(out_features=10),
            'a': dict(activation='linear')
        }
        return config

    def build_body(self, input_shape, config):
        """ Body block of densenet model. """
        transition_config = config.get('transition')
        dense_block = PeleeDenseBlock
        dense_config = config.get('dense')
        growth_rate = dense_config.get('growth_rate')
        bottleneck_factor = dense_config.get('bottleneck_factor')
        num_layers = dense_config.get('num_layers')
        kernel_size = dense_config.get('kernel_size')

        shape = input_shape
        body = Sequential()
        for i, (inum_layers, igrowth_rate) in enumerate(zip(num_layers,
                                                            growth_rate)):
            x = dense_block(
                input_shape=shape,
                block=self.conv_block,
                growth_rate=igrowth_rate,
                num_layers=inum_layers,
                kernel_size=kernel_size,
                bottleneck_factor=bottleneck_factor[i]
            )

            dense_transition = Sequential()
            dense_transition.add_module("DenseBlock", x)

            transition_layer = self.conv_block.partial(
                input_shape=x.output_shape,
                **transition_config
            )
            transition_layer = transition_layer(
                c=dict(filters=x.output_shape[0]))
            dense_transition.add_module("Transition",
                                        transition_layer)
            body.add_module("Block_{}".format(i),
                            dense_transition)
            shape = transition_layer.output_shape

        return body
