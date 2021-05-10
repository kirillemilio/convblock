""" Contains implementation of residual blocks. """

import numpy as np
from ..layers import ConvBlock
from ..bases import Identity
from ..bases import Module, MetaModule
from ..utils import transform_to_int_tuple
from .gcn import GCNBlock


class BaseResBlock(Module, metaclass=MetaModule):

    def __init__(self, input_shape, body, shortcut=None,
                 head=None, how='+', **kwargs):
        """ Build block with residual connection.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor.
        body : Module, ConvBlock, partially applied ConvBlock or None
            module that will be used for building body part of block.
            The only parameter of constructor must be input_shape.
            Can be partial module.
        shortcut : Module, ConvBlock, partially applied ConvBlock or None
            module that will be used for building shortcut part of block.
            The only parameter of constructor must be input_shape.
            Can be partial module. Default is None meaning that
            no head part required.
        how : str
            mode of merge operation. Default is '+'.
        """
        super().__init__(input_shape)
        self.body = body(input_shape=input_shape)
        self.shortcut = (Identity(input_shape) if shortcut is None
                         else shortcut(input_shape=input_shape))
        shortcut_out_shape = self.shortcut.output_shape
        if how == '.' and np.any(shortcut_out_shape[1:] != self.body.output_shape[1:]):
            raise ValueError("Output shape of shortcut "
                             + "must match output shape of body block.")
        elif how != '.' and np.any(shortcut_out_shape != self.body.output_shape):
            raise ValueError("Output shape of shortcut "
                             + "must match output shape of body block.")

        if how == '.':
            _shape = np.zeros_like(self.body.output_shape)
            _shape[:] = self.body.output_shape[:]
            _shape[0] += shortcut_out_shape[0]
        else:
            _shape = self.body.output_shape
        self.head = head if head is None else head(input_shape=_shape)
        self.how = how

    @property
    def output_shape(self) -> 'NDArray[int]':
        """ Get shape of the output tensor. """
        if self.head is not None:
            return self.head.output_shape
        elif self.how == '.':

            if self.shortcut is None:
                filters = self.body.output_shape[0] + self.input_shape[0]
            else:
                filters = self.body.output_shape[0] + \
                    self.shortcut.output_shape[0]

            spatial_dims = tuple(self.body.output_shape[1:])
            return np.array([filters, *spatial_dims], dtype=np.int)
        else:
            return self.body.output_shape

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for ResBlock. """
        x = self.body(inputs)
        y = inputs if self.shortcut is None else self.shortcut(inputs)
        z = self.merge(x, y, how=self.how)

        if self.head is None:
            return z
        else:
            return self.head(z)


class VanillaResBlock(BaseResBlock):

    def __init__(self, input_shape, filters, layout='cna cna c',
                 kernel_size=3, downsample=False,
                 post_activation=True, block=None, how='+', **kwargs):
        """ Build VanillaResBlock.

        VanillaResBlock is a generalization of convolutional block
        with residual connection used in ResNet.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor.
        filters : int, Tuple[int], List[int] or NDArray[int]
            number of output channels in the outputs of convolutional operations
            in body segment of block.
        layout : str
            layout or body part of ResBlock. Default is 'cna cna c'
        downsample : bool
            whether to set stride of first convolutional operation to 2.
            Default is False.
        kernel_size : int, Tuple[int], List[int] or NDArrray[int]
            kernel_sizes for convolutional operations in body part.
            Default is 3.
        post_activation : bool
            whether to use post activation with batch norm or not.
            Default is True.
        block : partially applied ConvBlock or None
            base block used for all convolutional operations under the hood.
        how : str
            merged method mode. Default is '+'.
        """
        block = ConvBlock if block is None else block
        num_conv = layout.lower().count('c')
        kernel_size = transform_to_int_tuple(kernel_size,
                                             'kernel_size',
                                             num_conv)

        filters = transform_to_int_tuple(filters, 'filters', num_conv)
        stride = (2 if downsample else 1, ) + (1, ) * (num_conv - 1)
        body = block.partial(
            layout=layout,
            c=dict(kernel_size=kernel_size,
                   filters=filters, stride=stride, **kwargs)
        )

        if downsample or (filters[-1] != input_shape[0] and how != '.'):
            shortcut = block.partial(
                layout='cn',
                c=dict(kernel_size=1,
                       stride=2 if downsample else 1,
                       filters=filters[-1])
            )
        else:
            shortcut = None

        if post_activation:
            head = block.partial(layout='an')
        else:
            head = None
        super().__init__(input_shape, body, shortcut, head, how)


class SimpleResBlock(VanillaResBlock):

    def __init__(self, input_shape, filters, kernel_size=3,
                 downsample=False, post_activation=True,
                 block=None, how='+', **kwargs):
        """ Build ResBlock without bottleneck.

        For more detatilts see https://arxiv.org/abs/1512.03385.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that batch dimension is not taken
            into account.
        filters : int
            number of channels in outputs of convolutional operations.
        kernel_size : int, Tuple[int], List[int] or NDArray[int]
            kernel_size for both convolutional operations. Default is 3.
        downsample : bool
            whether to set stride of first convolutional operation to 2.
            Default is False.
        post_activation : bool
            whether to use activation function after residuals were merged.
            Default is True.
        block : partially applied ConvBlock or None
            base block for all conv opeartions used under the hood.
        how : str
            merged method mode. Default is '+'.
        """
        layout = 'cna cn'
        kernel_size = transform_to_int_tuple(kernel_size, 'kernel_size', 2)
        filters = transform_to_int_tuple(filters, 'filters', 2)
        super().__init__(input_shape, filters, layout, kernel_size,
                         downsample, post_activation, block, how, **kwargs)


class BottleneckResBlock(VanillaResBlock):

    def __init__(self, input_shape, filters, kernel_size=(1, 3, 1),
                 factor=4, downsample=False, post_activation=True,
                 block=None, how='+', **kwargs):
        """ Build ResBlock with bottleneck.

        For more detatilts see https://arxiv.org/abs/1512.03385.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that batch dimension is not taken
            into account.
        filters : int
            number of channels in outputs of convolutional operations.
        kernel_size : int, Tuple[int], List[int] or NDArray[int]
            kernel_size for both convolutional operations. Default is 3.
        factor : int
            bottleneck factor that defines number of filters in the last
            convolutional layer: last_filters = filters * factor. Default is 4.
        downsample : bool
            whether to set stride of first convolutional operation to 2.
            Default is False.
        post_activation : bool
            whether to use activation function after residuals were merged.
            Default is True.
        block : partially applied ConvBlock or None
            base block for all conv opeartions used under the hood.
        how : str
            merged method mode. Default is '+'.
        """
        layout = 'cna cna cn'
        kernel_size = transform_to_int_tuple(kernel_size, 'kernel_size', 3)
        filters = (int(filters), int(filters), int(filters) * factor)
        super().__init__(input_shape, filters, layout, kernel_size,
                         downsample, post_activation, block, how, **kwargs)


class GCNResBlock(BaseResBlock, metaclass=MetaModule):

    def __init__(self, input_shape, filters, kernel_size=11, layout='cc',
                 post_activation=False, block=None, how='+'):
        """ Build Global Convolutional Block with residual connection.

        Implementation of global convolutional block with residual connection
        from http://arxiv.org/abs/1703.02719.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor.
        filters : int or Tuple[int]
            If int provided than transformed to (filters, filters) tuple.
            If tuple provided:
                filters[0] is the number of output channels
                for all convolutional operations inside GCN block.
                filters[1] is the number of output channels
                for shortcut of residual block.
        kernel_size: int
            size of kernel. Note that it must be int value, because
            1xk and kx1 convolutions are used under the hood.
        layout : str
            layout that is common for two branches. Must contain strictly two
            convolutional operation. Default is 'cc', but can be changed to
            'cna cna', 'cn cn' and etc.
        how : str
            the way to merge two branches. Can be '+', '.', '*'.
            Default is '+'. Note this value will be also used as parameter of
            ResBlock.
        block : ConvBlock or None
            partially applied ConvBlock or None. Default is None meaning
            that ConvBlock class itself will be used.
        post_activation : bool
            whether to use activation function after residuals were merged.
            Default is False.
        """
        block = ConvBlock if block is None else block
        filters = transform_to_int_tuple(filters, 'filters', 2)

        body = GCNBlock.partial(
            layout=layout,
            kernel_size=kernel_size,
            filters=filters[0], how=how,
            block=block
        )

        if filters[-1] != input_shape[0]:
            shortcut = block.partial(
                layout='c',
                c=dict(kernel_size=1, stride=1,
                       filters=filters[-1])
            )
        else:
            shortcut = None

        if post_activation:
            head = block.partial(layout='na')
        else:
            head = None
        super().__init__(input_shape, body, shortcut, head, how)
