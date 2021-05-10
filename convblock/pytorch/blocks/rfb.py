import numpy as np
import torch
from ..layers import ConvBlock
from ..bases import Module
from ..config import Config


class RFB(Module):

    @classmethod
    def default_config(cls):
        return Config({
            'branches': {
                0: {
                    'layout': 'cna cn',
                    'c': {
                        'kernel_size': [1, 3],
                        'dilation': [1, 1]
                    }
                },
                1: {
                    'layout': 'cna cna cn',
                    'c': {
                        'kernel_size': [1, 3, 3],
                        'dilation': [1, 1, 3]
                    }
                },
                2: {
                    'layout': 'cna cna cna cn',
                    'c': {
                        'kernel_size': [1, 3, 3, 3],
                        'dilation': [1, 1, 1, 5]
                    }
                }
            },
            'shortcut': {
                'layout': 'cn',
                'c': {
                    'kernel_size': 1
                }
            },
            'post': {
                'layout': 'cn',
                'c': {
                    'kernel_size': 1
                }
            },
            'head': {
                'layout': 'a',
                'a': {
                    'activation': 'relu'
                }
            }
        })

    def build_config(self, config,
                     out_filters=None,
                     branches_filters=None,
                     downsample=False):
        out_filters = (self.input_shape[0]
                       if out_filters is None
                       else int(out_filters))
        if branches_filters is None:
            channels = self.in_channels // 8
            branches_filters = [
                2 * channels,
                (channels, 2 * channels, 2 * channels),
                (channels, 3 * (channels // 2),
                 2 * channels, 2 * channels)
            ]
        stride = 2 if downsample else 1
        branches_config = Config({
            0: {'c/filters': branches_filters[0], 'c/stride': [1, stride]},
            1: {'c/filters': branches_filters[1], 'c/stride': [1, stride, 1]},
            2: {'c/filters': branches_filters[2], 'c/stride': [1, 1, stride, 1]}
        })
        return config @ {
            'branches': branches_config,
            'post': {'c/filters': out_filters},
            'shortcut': {'c/filters': out_filters, 'c/stride': stride},
            'head': {'c/filters': out_filters}
        }

    def __init__(self, input_shape,
                 out_filters=None,
                 branches_filters=None,
                 downsample: bool = False,
                 config: dict = None):
        super().__init__(input_shape)
        config = self.default_config() @ Config(config)
        config = self.build_config(config,
                                   out_filters,
                                   branches_filters,
                                   downsample)
        branches_config = config.get('branches')
        self.branches = torch.nn.ModuleList()
        shape = np.array([0, 0, 0])
        for i in range(len(branches_config)):
            branch = ConvBlock(
                input_shape=self.input_shape,
                **branches_config[i]
            )
            shape[0] += branch.output_shape[0]
            shape[1:] = branch.output_shape[1:]
            self.branches.append(branch)

        if config.get('post'):
            self.post = ConvBlock(
                input_shape=shape,
                **config['post']
            )
            shape = self.post.output_shape
        else:
            self.post = None

        if self.input_shape[0] != out_filters or downsample:
            self.shortcut = ConvBlock(
                input_shape=input_shape,
                **config['shortcut']
            )

        else:
            self.shortcut = None

        if config.get('head'):
            self.head = ConvBlock(
                input_shape=shape,
                **config['head']
            )
            shape = self.head.output_shape
        else:
            self.head = None

        self._output_shape = shape

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x):
        outputs = []
        for branch in self.branches:
            outputs.append(branch(x))

        y = torch.cat(outputs, 1)
        if self.post:
            y = self.post(y)
        if self.shortcut:
            y = y + self.shortcut(x)
        else:
            y = y + x
        return (y if self.head is None
                else self.head(y))


class RFBa(RFB):

    @classmethod
    def default_config(cls):
        return RFB.default_config() @ {
            'branches': {
                0: {
                    'layout': 'cna cn',
                    'c': {
                        'kernel_size': [1, 3],
                        'dilation': [1, 1]
                    }
                },
                1: {
                    'layout': 'cna cna cn',
                    'c': {
                        'kernel_size': [(1, 1),
                                        (3, 1),
                                        (3, 3)],
                        'dilation': [1, 1, 3]
                    }
                },
                2: {
                    'layout': 'cna cna cn',
                    'c': {
                        'kernel_size': [(1, 1),
                                        (1, 3),
                                        (3, 3)],
                        'dilation': [1, 1, 3]
                    }
                },
                3: {
                    'layout': 'cna cna cna cn',
                    'c': {
                        'kernel_size': [(1, 1),
                                        (1, 3),
                                        (3, 1),
                                        (3, 3)],
                        'dilation': [1, 1, 1, 5]
                    }
                }
            }
        }

    def build_config(self, config,
                     out_filters=None,
                     branches_filters=None,
                     downsample=False):
        if downsample:
            raise NotImplementedError("Downsample is not supported"
                                      + " by RBFa module")
        out_filters = (self.input_shape[0]
                       if out_filters is None
                       else int(out_filters))
        if branches_filters is None:
            channels = self.in_channels // 8
            branches_filters = [
                2 * channels,
                (2 * channels, 2 * channels, 2 * channels),
                (channels, 3 * (channels // 2),
                 2 * channels, 2 * channels)
            ]
        branches_config = Config({
            0: {'c/filters': branches_filters[0]},
            1: {'c/filters': branches_filters[1]},
            2: {'c/filters': branches_filters[1]},
            3: {'c/filters': branches_filters[2]}
        })
        return config @ {
            'branches': branches_config,
            'post': {'c/filters': out_filters},
            'shortcut': {'c/filters': out_filters},
            'head': {'c/filters': out_filters}
        }


class LiteRFBa(RFB):

    @classmethod
    def default_config(cls):
        return RFB.default_config() @ {
            'branches': {
                0: {
                    'layout': 'cna cn',
                    'c': {
                        'kernel_size': [1, 3],
                        'dilation': [1, 1]
                    }
                },
                1: {
                    'layout': 'cna cna cn',
                    'c': {
                        'kernel_size': [(1, 1),
                                        (3, 1),
                                        (3, 3)],
                        'dilation': [1, 1, 3]
                    }
                },
                2: {
                    'layout': 'cna cna cn',
                    'c': {
                        'kernel_size': [(1, 1),
                                        (1, 3),
                                        (3, 3)],
                        'dilation': [1, 1, 3]
                    }
                },
                3: {
                    'layout': 'cna cna cna cn',
                    'c': {
                        'kernel_size': [(1, 1),
                                        (1, 3),
                                        (3, 1),
                                        (3, 3)],
                        'dilation': [1, 1, 1, 5]
                    }
                }
            }
        }

    def build_config(self, config,
                     out_filters=None,
                     branches_filters=None,
                     downsample=False):
        if downsample:
            raise NotImplementedError("Downsample is not supported"
                                      + " by RBFa module")
        out_filters = (self.input_shape[0]
                       if out_filters is None
                       else int(out_filters))
        branches_config = config['branches']
        conv_counts = [
            branches_config[i]['layout'].count('c')
            for i in range(len(branches_config))
        ]
        if branches_filters is None:
            channels = self.in_channels // 8
            branches_filters = [
                2 * channels,
                (2 * channels, 2 * channels, 2 * channels),
                (2 * channels, 2 * channels, 2 * channels),
                (channels, 3 * (channels // 2),
                 2 * channels, 2 * channels)
            ]
            branches_groups = [
                [1] * (conv_counts[0] - 1) + [2 * channels],
                [1] * (conv_counts[1] - 1) + [2 * channels],
                [1] * (conv_counts[2] - 1) + [2 * channels],
                [1] * (conv_counts[3] - 1) + [2 * channels],
            ]
        else:
            branches_groups = [
                [1] * conv_counts[0],
                [1] * conv_counts[1],
                [1] * conv_counts[2],
                [1] * conv_counts[3]
            ]
        branches_config = Config({
            0: {'c/filters': branches_filters[0],
                'c/groups': branches_groups[0]},
            1: {'c/filters': branches_filters[1],
                'c/groups': branches_groups[1]},
            2: {'c/filters': branches_filters[2],
                'c/groups': branches_groups[2]},
            3: {'c/filters': branches_filters[3],
                'c/groups': branches_groups[3]}
        })
        return config @ {
            'branches': branches_config,
            'post': {'c/filters': out_filters},
            'shortcut': {'c/filters': out_filters},
            'head': {'c/filters': out_filters}
        }


class LiteRFB(RFB):

    @classmethod
    def default_config(cls):
        return RFB.default_config() @ {
            'branches': {
                0: {
                    'layout': 'cna cna cna cn',
                    'c': {
                        'kernel_size': [(1, 1),
                                        (1, 3),
                                        (3, 1),
                                        (3, 3)],
                        'dilation': [1, 1, 1, 3]
                    }
                },
                1: {
                    'layout': 'cna cna cna cn',
                    'c': {
                        'kernel_size': [(1, 1),
                                        (3, 3),
                                        (3, 3),
                                        (3, 3)],
                        'dilation': [1, 1, 1, 5]
                    }
                }
            }
        }

    def build_config(self, config,
                     out_filters=None,
                     branches_filters=None,
                     downsample=False):
        if downsample:
            raise NotImplementedError("Downsample is not supported"
                                      + " by RBFa module")
        out_filters = (self.input_shape[0]
                       if out_filters is None
                       else int(out_filters))
        branches_config = config['branches']
        conv_counts = [
            branches_config[i]['layout'].count('c')
            for i in range(len(branches_config))
        ]
        if branches_filters is None:
            channels = self.in_channels // 8
            branches_filters = [
                [channels, 3 * (channels // 2), 3 *
                 (channels // 2), 3 * (channels // 2)],
                [channels, 3 * (channels // 2), 3 *
                 (channels // 2), 3 * (channels // 2)]
            ]
            branches_groups = [
                [1] * (conv_counts[0] - 1) + [3 * (channels // 2)],
                [1] * (conv_counts[1] - 1) + [3 * (channels // 2)]
            ]
        else:
            branches_groups = [
                [1] * conv_counts[0],
                [1] * conv_counts[1],
            ]
        branches_config = Config({
            0: {'c/filters': branches_filters[0],
                'c/groups': branches_groups[0]},
            1: {'c/filters': branches_filters[1],
                'c/groups': branches_groups[1]}
        })
        return config @ {
            'branches': branches_config,
            'post': {'c/filters': out_filters},
            'shortcut': {'c/filters': out_filters},
            'head': {'c/filters': out_filters}
        }
