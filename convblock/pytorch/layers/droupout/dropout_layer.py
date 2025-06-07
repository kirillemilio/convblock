"""Contains implementation of dropout layer."""

from __future__ import annotations

from typing import Literal

import torch

from ...bases import Layer
from ...utils import ArrayLike
from ..conv_block import ConvBlock
from .dropblock import DropBlock1D, DropBlock2D, DropBlock3D


@ConvBlock.register_option(name="d")
class DropoutLayer(Layer):
    """Dropout layer option implementation.

    Attributes
    ----------
    input_shape : ArrayLike[int]
        input shape of dropout layer.
        Note that batch dimension is not taken into consideration.
    p : float
        probability of dropout or dropblock.
    mode : Literal["pixels", "channels", "dropblock"]
        dropout model: pixels(random noise), dropout by channels(classical dropout)
        or dropblock. Default is "channels".
    inplace : bool
        whether to to apply dropout inplace. Note that
        this option will be ignored in case of mode being "dropblock".
        Default is True.
    block_size : int
        block_size in case of mode being "dropblock".
        Default is 1.
    """

    p: float
    mode: Literal["pixels", "channels", "dropblock"]

    def __init__(
        self,
        input_shape: ArrayLike[int],
        p: float = 0.5,
        inplace: bool = True,
        block_size: int = 1,
        mode: Literal["pixels", "channels", "blocks"] = "channels",
    ) -> None:
        """
        Initialize a dropout or dropblock layer depending on mode and input.

        Parameters
        ----------
        input_shape : ArrayLike[int]
            Shape of input tensor excluding batch dimension.
        p : float, optional
            Drop probability. Defines how much of the input will be dropped.
            Default is 0.5.
        inplace : bool, optional
            Whether to apply dropout in-place. Ignored when mode is "blocks".
            Default is True.
        block_size : int, optional
            Size of the square/cubic block to drop when using "blocks" mode.
            Default is 1.
        mode : {"pixels", "channels", "blocks"}, optional
            Dropout mode:
            - "pixels": Apply dropout to individual elements (Dropout).
            - "channels": Apply dropout to full channels (DropoutNd).
            - "blocks": Apply DropBlock (structured dropout by regions).
            Default is "channels".

        Raises
        ------
        NotImplementedError
            If the input dimension or mode combination is not supported.
        """
        ndim = len(input_shape[1:])
        self.p = p
        self.mode = mode

        match mode:
            case "pixels":
                layer = torch.nn.Dropout(p, inplace=inplace)
            case "channels" if ndim == 1:
                layer = torch.nn.Dropout1d(p, inplace=inplace)
            case "channels" if ndim == 2:
                layer = torch.nn.Dropout2d(p, inplace=True)
            case "channels" if ndim == 3:
                layer = torch.nn.Dropout3d(p, inplace=inplace)
            case "blocks" if ndim == 1:
                layer = DropBlock1D(proba=p, block_size=block_size)
            case "blocks" if ndim == 2:
                layer = DropBlock2D(proba=p, block_size=block_size)
            case "blocks" if ndim == 3:
                layer = DropBlock3D(proba=p, block_size=block_size)
            case _:
                raise NotImplementedError(
                    "Dropout mode must be one of `pixels`, `channels`, `blocks`. "
                    + "Number of dimensions must be one of (1, 2, 3)."
                )
        super().__init__(input_shape=input_shape, layer=layer)
