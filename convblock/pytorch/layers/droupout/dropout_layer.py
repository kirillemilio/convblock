"""Contains implementation of dropout layer."""

from __future__ import annotations

from typing import Literal

import torch

from ...utils import ArrayLike
from ..conv_block import ConvBlock
from ..torch_module import TorchModule
from .dropblock import DropBlock1D, DropBlock2D, DropBlock3D


@ConvBlock.register_option(name="d")
class DropoutLayer(TorchModule):
    """Dropout layer option implementation.

    Attributes
    ----------
    p : float
        probability of dropout or dropblock.
    mode : Literal["pixels", "channels", "dropblock"]
        dropout model: pixels(random noise), dropout by channels(classical dropout)
        or dropblock. Default is "channels".
    layer : torch.nn.Module
        underlying module.
    """

    p: float
    mode: Literal["pixels", "channels", "dropblock"]
    layer: torch.nn.Module

    def __init__(
        self,
        input_shape: ArrayLike[int],
        p: float = 0.5,
        inplace: bool = True,
        block_size: int = 1,
        mode: Literal["pixels", "channels", "dropblock"] = "channels",
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
            Whether to apply dropout in-place. Ignored when mode is "dropblock".
            Default is True.
        block_size : int, optional
            Size of the square/cubic block to drop when using "dropblock" mode.
            Default is 1.
        mode : {"pixels", "channels", "dropblock"}, optional
            Dropout mode:
            - "pixels": Apply dropout to individual elements (Dropout).
            - "channels": Apply dropout to full channels (DropoutNd).
            - "dropblock": Apply DropBlock (structured dropout by regions).
            Default is "channels".

        Raises
        ------
        NotImplementedError
            If the input dimension or mode combination is not supported.
        """
        super().__init__(input_shape=input_shape, output_shape=input_shape)
        ndim = len(input_shape) - 1
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
            case "dropblock" if ndim == 1:
                layer = DropBlock1D(proba=p, block_size=block_size)
            case "dropblock" if ndim == 2:
                layer = DropBlock2D(proba=p, block_size=block_size)
            case "dropblock" if ndim == 3:
                layer = DropBlock3D(proba=p, block_size=block_size)
            case _:
                raise NotImplementedError(
                    "Dropout mode must be one of `pixels`, `channels`, `dropblock`. "
                    + "Number of dimensions must be one of (1, 2, 3)."
                )
        self.layer = layer

    def forward(self, inputs: torch.Tensor, *others: torch.Tensor) -> torch.Tensor:
        """Forward torch tensors through module.

        Parameters
        ----------
        *inputs : torch.Tensor
            input tensors.

        Returns
        -------
        list[torch.Tensor]
            list of output tensors.
        """
        return self.layer(inputs)
