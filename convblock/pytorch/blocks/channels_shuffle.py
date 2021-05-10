""" Contains implementation of Shuffle Units used in ShuffleNet v1 and v2 architectures. """

import math
import numpy as np
import torch
from ..layers import ConvBlock
from ..utils import transform_to_int_tuple, INT_TYPES, FLOAT_TYPES
from .res_block import BaseResBlock


class CShuffleUnit(BaseResBlock):

    def __init__(self, input_shape, filters, layout='cna s cn cn',
                 kernel_size=(1, 3, 1), factor=4, groups=4,
                 downsample=False, pool_kernel=3, pool_mode='avg',
                 post_activation=True, block=None):
        """ Build residual block with channels shuffle.

        Main building block of ShuffleNet architecture.
        For more detailts see https://arxiv.org/pdf/1707.01083.pdf.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor.
        filters : int
            number of filters in the output of layer.
        layout : str
            layout must contain exactly 3 convolution operations.
        kernel_size : int, Tuple[int], List[int] or NDArray[int]
            kernel_size for convolutional operations. Default is 3.
        factor : int
            bottleneck factor that defines number of filters in first two
            convolution layers as filters / factor. Default is 4.
        groups: int or ArrayLike[int] of size 2
            number of groups in first and third convolution operations.
            Default is 4 for both convolutions.
        downsample: bool
            whether to use downsampling. Default is False.
        pool_kernel: int, Tuple[int], List[int] or NDArray[int]
            kernel_size of pooling operation. Default is 3.
        pool_mode : str
            type of pooling operation. Can be 'avg' or 'max'. Default is 'avg'.
        block : ConvBlock or None
            partially applied ConvBlock or None. Default is None meaning
            that ConvBlock class itself will be used.
        post_activation : bool
            whether to use activation function after residuals were merged.
            Default is False.
        """
        block = ConvBlock if block is None else block
        num_conv = layout.lower().count('c')

        if not isinstance(filters, INT_TYPES):
            raise ValueError("Argument 'filters' must have integer type")
        else:
            filters = int(filters)

        if num_conv != 3:
            raise ValueError("Number of convolution operations"
                             + " in layout must be equal to 3.")

        kernel_size = transform_to_int_tuple(kernel_size,
                                             'kernel_size',
                                             num_conv)

        groups = transform_to_int_tuple(groups, 'groups', 2)
        filters = filters - input_shape[0] if downsample else filters
        stride = (1, 2 if downsample else 1) + (num_conv - 2) * (1, )
        body = block.partial(
            layout=layout,
            c=dict(kernel_size=kernel_size,
                   groups=(groups[0], filters // factor, groups[1]),
                   filters=(filters // factor, filters // factor, filters),
                   stride=stride)
        )

        if downsample:
            shortcut = block.partial(
                layout='p',
                p=dict(kernel_size=pool_kernel, stride=2, mode=pool_mode)
            )
            how = '.'
        else:
            shortcut = None
            how = '+'

        if post_activation:
            head = block.partial(layout='a')
        else:
            head = None
        super().__init__(input_shape, body, shortcut, head, how)


class CSplitAndShuffleUnit(BaseResBlock):

    def __init__(self, input_shape, filters, share=0.5,
                 downsample=False, block=None):
        """ Build residual block with channels shuffle.

        Main building block of ShuffleNet architecture.
        For more detailts see https://arxiv.org/pdf/1707.01083.pdf.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor.
        filters : int
            number of channels in the output tensor.
        share : float
            share of of channels that will be passed through body
            if 'downsample' parameter is set to False. Default is 0.5.
        downsample : bool
            whether to use downsampling. Default is False.
        block : ConvBlock or None
            partially applied ConvBlock or None. Default is None meaning
            that ConvBlock class itself will be used.
        """

        if not isinstance(share, FLOAT_TYPES):
            raise TypeError("Argument 'share' must be float.")

        share = float(share)
        if share < 0 or share > 1:
            raise ValueError("Argument 'share' must be float"
                             + " from [0, 1] interval. Got {}.".format(share))

        block = ConvBlock if block is None else block

        if downsample:
            body_filters = filters - input_shape[0]
            shortcut_filters = input_shape[0]
        else:
            if int(filters) != int(input_shape[0]):
                raise ValueError("Argument 'filters' must be equal to number"
                                 + " of input channels in case"
                                 + " 'downsample' is set to False.")

            body_filters = math.floor(share * filters)
            shortcut_filters = filters - body_filters

        body = block.partial(
            layout='cna cn cna' if downsample else 'l cna cn cna',
            c=dict(kernel_size=[1, 3, 1],
                   groups=(1, body_filters, 1),
                   filters=body_filters,
                   stride=(1, 2 if downsample else 1, 1)),
            l=dict(op=lambda x: x[:, :body_filters, ...],
                   annotation='x => x[:, :{}, ...]'.format(body_filters),
                   output_shape=np.array([body_filters, *input_shape[1:]]))
        )
        shortcut = block.partial(
            layout='cn cna' if downsample else 'l',
            c=dict(filters=shortcut_filters, kernel_size=(3, 1),
                   stride=(2, 1), groups=(shortcut_filters, 1)),
            l=dict(op=lambda x: x[:, body_filters:, ...],
                   annotation='x => x[:, {}:, ...]'.format(body_filters),
                   output_shape=np.array([shortcut_filters, *input_shape[1:]]))
        )
        head = block.partial(layout='s')
        super().__init__(input_shape, body, shortcut, head, how='.')
