import math
import numpy as np
import torch
import torch.nn.functional as F

from ..layers import ConvBlock
from ..bases import Module
from .res_block import VanillaResBlock


class BaseDecoder(Module):

    def __init__(self, input_shape, skip_shape,
                 input_block=None, skip_block=None,
                 body=None, how='+'):
        """ Base class for all decoder blocks.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that
            batch dimension is not taken in account.
        skip_shape : Tuple[int], List[int] or NDAarray[int]
            shape of the skip tensor. Note that
            batch dimension is not taken in account.
        input_block : ConvBlock, partially applied ConvBlock or None
            block of operations that will be applied to input tensor.
            If None then no operation will be applied. Default is None.
        skip_block : ConvBlock, partially applied ConvBlock or None
            block of operations that will be applied to skip tensor.
            If None then no operations will be applied. Default is None.
        body : ConvBlock, partially applied ConvBlock or None
            block of operations that will be applied after
            outputs of input_block and skip_block will be merged
            (see 'how' argument that defines type of operation
            that will be used merging). If None then no operations
            will be applied. Default is None.
        how : str
            '+', '.' or '*'. Type of operation that will be used for merging.
            Default is '+'.
        """

        super().__init__(input_shape)
        self._skip_shape = skip_shape
        self.skip_block = skip_block(
            input_shape=skip_shape) if skip_block else None
        self.input_block = input_block(
            input_shape=input_shape) if input_block else None
        self.how = how

        shape = np.zeros_like(self.input_shape, dtype=np.int)
        sout_shape = self.skip_block.output_shape if skip_block else self.skip_shape
        iout_shape = self.input_block.output_shape if input_block else self.input_shape
        if self.how in ('+', '*'):
            # assert np.all(iout_shape == sout_shape)
            shape[0] = self.input_block.output_shape[0]
        elif self.how == '.':
            # assert np.all(iout_shape[1:] == sout_shape[1:])
            shape[0] = iout_shape[0] + sout_shape[0]
        else:
            raise ValueError("Argument 'how' must have "
                             + "one of following values: ('+', '.', '*')")

        shape[1:] = iout_shape[1:]
        self.body = body(input_shape=shape) if body else None
        self._output_shape = self.body.output_shape if self.body else shape

    @property
    def skip_shape(self):
        """ Get shape of skip-connection tensor.

        Returns
        -------
        NDArray[int]
            shape of the output tensor.
        """
        return np.array(self._skip_shape, np.int)

    @property
    def output_shape(self):
        """ Get shape of the output tensor.

        Returns
        -------
        NDArray[int]
            shape of the output tensor.
        """
        return np.array(self._output_shape, np.int)

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for decoder. """
        prev, skip = inputs
        x = self.input_block(prev) if self.input_block else prev
        y = self.skip_block(skip) if self.skip_block else skip
        y = self.crop_as(y, x)

        if self.how == '+':
            z = x + y
        elif self.how == '.':
            z = torch.cat([x, y], dim=1)
        elif self.how == '*':
            z = x * y

        if self.body:
            return self.body(z)
        else:
            return z


class VanillaUNetDecoder(BaseDecoder):

    def __init__(self, input_shape, skip_shape, filters, layout, kernel_size=3,
                 upsampling_mode='t', upsampling_kernel=2, block=None):
        """ Encoder for vanilla UNet architecture.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that
            batch dimension is not taken in account.
        skip_shape : Tuple[int], List[int] or NDAarray[int]
            shape of the skip tensor. Note that
            batch dimension is not taken in account.
        filters : ArrayLike[int] or int
            filters for convolutions in ConvBlock.
        layout : str
            layout of ConvBlock or partially applied ConvBlock.
        kernel_size : ArrayLike[int] or int
            kernel_size required by convolutions of ConvBlock.
            Default is 3 meaning that all convolutions inside block along
            each spatial dimension will have kernel_size equal to 3.
        upsampling_mode : str
            upsampling mode. 't' stands for transposed convolution
            while 'u' stands for upsampling.
            Default is 't'.
        upsampling_kernel : ArrayLike[int] or int
            kernel_size of transposed convolution used for upsampling.
            Default is 2.
        block : ConvBlock, partially applied ConvBlock or None.
            this argument can be used for passing non-default
            global parameters to ConvBlock.
        """
        block = ConvBlock if block is None else block

        if upsampling_mode == 't':
            upsampling_layout = 'tna'
        elif upsampling_mode == 'u':
            upsampling_layout = 'uc'
        else:
            raise ValueError("Argument 'upsampling_mode' must be one of"
                             + " following values: 'u'"
                             + " or 't'. Got '{}'.".format(upsampling_mode))

        input_block = block.partial(
            layout=upsampling_layout,
            u=dict(scale=2, mode=upsampling_mode),
            c=dict(kernel_size=1, filters=filters),
            t=dict(kernel_size=upsampling_kernel,
                   filters=filters, stride=2)
        )

        body = block.partial(
            layout='cna cna',
            c=dict(kernel_size=kernel_size,
                   filters=filters)
        )

        super().__init__(input_shape, skip_shape, input_block, None, body, '.')


class VanillaVNetDecoder(BaseDecoder):

    def __init__(self, input_shape, skip_shape, filters, layout, kernel_size=5,
                 upsampling_mode='t', upsampling_kernel=2,
                 post_activation=False, block=None):
        """ Encoder for vanilla UNet architecture.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that
            batch dimension is not taken in account.
        skip_shape : Tuple[int], List[int] or NDAarray[int]
            shape of the skip tensor. Note that
            batch dimension is not taken in account.
        filters : ArrayLike[int] or int
            filters for convolutions in ConvBlock.
        layout : str
            layout of ConvBlock or partially applied ConvBlock.
        kernel_size : ArrayLike[int] or int
            kernel_size required by convolutions of ConvBlock.
            Default is 3 meaning that all convolutions inside block along
            each spatial dimension will have kernel_size equal to 3.
        upsampling_mode : str
            upsampling mode. 't' stands for transposed convolution
            while 'u' stands for upsampling.
            Default is 't'.
        upsampling_kernel : ArrayLike[int] or int
            kernel_size of transposed convolution used for upsampling.
            Default is 2.
        post_activation : bool
            whether to use post activation with normalization
            in VanillaResBlock. Default is False.
        block : ConvBlock, partially applied ConvBlock or None.
            this argument can be used for passing non-default
            global parameters to ConvBlock.
        """
        block = ConvBlock if block is None else block

        if upsampling_mode == 't':
            upsampling_layout = 'tna'
        elif upsampling_mode == 'u':
            upsampling_layout = 'uc'
        else:
            raise ValueError("Argument 'upsampling_mode' must be one of"
                             + " following values: 'u'"
                             + " or 't'. Got '{}'.".format(upsampling_mode))

        input_block = block.partial(
            layout=upsampling_layout,
            u=dict(scale=2, mode=upsampling_mode),
            c=dict(kernel_size=1, filters=filters),
            t=dict(kernel_size=upsampling_kernel,
                   filters=filters, stride=2)
        )

        body = VanillaResBlock.partial(
            filters=filters, layout=layout,
            kernel_size=kernel_size, block=block,
            post_activation=post_activation
        )
        super().__init__(input_shape, skip_shape, input_block, None, body, '.')
