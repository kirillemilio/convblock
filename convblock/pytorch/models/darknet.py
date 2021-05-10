""" Contains implementation of DarkNet architecture. """
from typing import Tuple, List, Dict, Set, Any, Optional
from .base_model import BaseModel
from ..bases import Sequential


class DarkNet19(BaseModel):

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()

        config['body'] = {
            'filters': (32, 64, 128,
                        256, 512, 1024),
            'num_blocks': (0, 0, 1,
                           1, 2, 2),
            'downsample': [True, True, True,
                           True, True, False],
            'pool_size': 3,
            'pool_mode': 'max'
        }

        config['head'] = {
            'layout': 'cna > a',
            'c': dict(filters=1000, kernel_size=1),
            'a': dict(activation=('relu', 'softmax'))
        }
        return config

    def build_body(self, input_shape, config):
        filters = config.get('filters')
        num_blocks = config.get('num_blocks')
        pool_size = config.get('pool_size')
        pool_mode = config.get('pool_mode')
        downsample = config.get('downsample')
        body = Sequential()
        shape = input_shape
        for i, (f, b) in enumerate(zip(filters, num_blocks)):
            layout = 'cna cna' * b + 'cna' + ('p' if downsample[i] else '')
            ifilters = [f, f // 2] * b + [f]
            kernel_size = [3, 1] * b + [3]
            x = self.conv_block(
                input_shape=shape,
                layout=layout,
                c=dict(filters=ifilters,
                       kernel_size=kernel_size),
                p=dict(kernel_size=pool_size,
                       stride=2, mode=pool_mode)
            )
            body.add_module(f'Block_{i}', x)
            shape = x.output_shape
        return body


class DarkNet53(BaseModel):

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()

        config['conv_block'] = {
            'a': {
                'activation': 'leaky_relu',
                'negative_slope': 0.1,
                'inplace': True
            }
        }

        config['input'] = {
            'layout': 'cna',
            'c': {
                'filters': 32,
                'kernel_size': 3
            }
        }

        config['body'] = {
            'filters': (64, 128, 256,
                        512, 1024),
            'num_blocks': (1, 2, 8, 8, 4),
            'downsample': [True, True, True,
                           True, True]
        }

        config['head'] = {
            'filters': (256, 256, 256),
            'head_filters': [(255, None),
                             (255, 256),
                             (255, 256)]
        }
        return config

    @classmethod
    def rout_block(cls, input_shape, filters,
                   head_filters=(255, 256), block=None):
        conv_block = ConvBlock if block is None else block
        body = conv_block(
            input_shape=input_shape,
            layout='cna cna cna cna cna',
            c=dict(kernel_size=[1, 3, 1, 3, 1],
                   filters=[filters // 2, filters,
                            filters // 2, filters,
                            filters // 2])
        )
        main_branch = conv_block(
            input_shape=body.output_shape,
            layout='cna cna',
            c=dict(kernel_size=[3, 1],
                   filters=(filters, head_filters[0]))
        )
        if head_filters[1] is None:
            return Sequential(body, main_branch)
        branches = Branches([
            main_branch,
            conv_block(
                input_shape=body.output_shape,
                layout='cna u',
                c=dict(kernel_size=1,
                       filters=head_filters[1]),
                u=dict(scale=2)
            )
        ], mode=None)
        return Sequential(body, branches)

    @classmethod
    def block(cls,
              input_shape: Tuple[int, ...],
              filters: int,
              downsample: bool = False,
              num_blocks: int = 1,
              block: Any = None):
        """ Build main res-block of DarkNet53 architecture used in body. """
        conv_block = ConvBlock if block is None else block
        if downsample:
            kernel_size = [3] + [1, 3] * num_repeats
            stride = [2] + [1, 1] * num_repeats
            layout = 'cna' + '+ cna cn + a' * num_repeats
            filters = [filters] + [filters // 2, filters] * num_repeats
        else:
            kernel_size = [1, 3] * num_repeats
            stride = [1, 1] * num_repeats
            layout = '+ cna cn + a' * num_repeats
            filters = [filters // 2, filters] * num_repeats
        return conv_block(
            input_shape=input_shape,
            layout=layout,
            c=dict(kernel_size=kernel_size,
                   filters=filters,
                   stride=stride),
            shortcut=dict(allow_identity=False,
                          layout='cn')
        )

    def build(self, *args, **kwargs):
        config = self.config
        input_shape = config.get('input_shape')
        self.input = self.build_input(input_shape=input_shape,
                                      config=config.get('input'))
        if self.input is not None:
            input_shape = self.input.output_shape

        self.body = self.build_body(input_shape=input_shape,
                                    config=config.get('body'))

        self.head = self.build_head(input_shape=None,
                                    config=config.get('head'))

        self.output_shape = None

    def build_body(self, input_shape, config):
        filters = config.get('filters')
        num_blocks = config.get('num_blocks')
        downsample = config.get('downsample')
        shape = input_shape
        body = Sequential()
        for i, ifilters in enumerate(filters):
            x = self.block(
                input_shape=shape,
                filters=ifilters,
                num_blocks=num_blocks[i],
                downsample=downsample[i],
                block=self.conv_block
            )
            body.add_module("Block_{}".format(i), x)
            shape = x.output_shape
        return body

    def build_head(self, input_shape, config):
        filters = config.get('filters')
        head_filters = config.get('head_filters')
        head_layers = torch.nn.ModuleList()
        shape = [0, 0, 0]
        for i, (f, h) in enumerate(zip(filters[::-1],
                                       head_filters[::-1])):
            _shape = [shape[0] + self.body[-1-i].output_shape[0],
                      *self.body[-1-i].output_shape[1:]]
            x = self.rout_block(_shape, f, h,
                                self.conv_block)
            out_shape = np.atleast_2d(x.output_shape)
            head_layers.append(x)
            if out_shape.shape[0] > 1:
                shape = out_shape[1, :]
            else:
                shape = None
        return head_layers[::-1]

    def forward(self, x):
        y = self.input(x)

        body_outputs = []
        for i, module in enumerate(self.body):
            y = module(y)
            body_outputs.append(y)

        head_outputs = []
        up = None
        for y, head in zip(body_outputs[::-1],
                           self.head[::-1]):
            if up is None:
                z, up = head(y)
            else:
                z, up = head(torch.cat([y, up], 1))
            head_outputs.append(z)

        return head_outputs[::-1]
