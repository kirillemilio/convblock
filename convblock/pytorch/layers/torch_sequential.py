"""Contains implementation of simple pytorch wrapped module."""

from __future__ import annotations

import torch

from .base_module import BaseModule
from .torch_module import TorchModule


class TorchSequential(BaseModule, torch.nn.Sequential):
    """Provides implementation of pytorch module wrapper."""

    def __init__(self, arg: TorchModule, *args: TorchModule) -> None:
        """Initialize simple module with specified input and output shapes.

        Parameters
        ----------
        input_shape : NDArray[np.int64]
            1D or 2D numpy array defining expected shapes of input tensors.
        output_shape : NDArray[np.int64]
            1D or 2D numpy array defining expected shapes of output tensors.

        Raises
        ------
        ValueError
            If eighter input_shape or output_shape has more than 2 dimensions.
        """
        input_shape = arg.input_shape.copy()
        last_module = arg
        for module in args:
            last_module = module
        output_shape = last_module.output_shape.copy()
        torch.nn.Sequential.__init__(self, arg, *args)
        BaseModule.__init__(self, input_shape=input_shape, output_shape=output_shape)

    def forward(self, inputs: torch.Tensor, *others: torch.Tensor) -> list[torch.Tensor]:
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
        for module in self:
            inputs = module(inputs)
        return inputs

    def count_parameters(self, include_static: bool = False) -> int:
        """Count number of parameters.

        Parameters
        ----------
        include_static : bool
            whether to include static(non trainable parameters)
            Default is False meaning that only parameters
            requiring grad will be taken into consideration.

        Returns
        -------
        int
            number of parameters
            requiring gradient computation.
        """
        if include_static:
            return sum(p.numel() for p in self.parameters())
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
