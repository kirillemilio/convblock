"""Contains implementation of base module class."""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from ..utils import crop_as, merge, pad_as
from .module_interface import IModule


class BaseModule(IModule):
    """
    Provide a base implementation for custom neural network modules.

    This class stores input and output shapes for each tensor and extends both
    a PyTorch module and a custom interface. It includes utility methods for
    accessing tensor shapes, counting parameters, and handling tensor
    broadcasting operations like padding, cropping, and merging.
    """

    input_shape: NDArray[np.int64]
    output_shape: NDArray[np.int64]

    def __init__(self, input_shape: NDArray[np.int64], output_shape: NDArray[np.int64]) -> None:
        """
        Initialize BaseModule with specified input and output shapes.

        Parameters
        ----------
        input_shape : NDArray[np.int64]
            1D or 2D numpy array defining expected shapes of input tensors.
        output_shape : NDArray[np.int64]
            1D or 2D numpy array defining expected shapes of output tensors.

        Raises
        ------
        ValueError
            If either input_shape or output_shape has more than 2 dimensions.
        """
        super().__init__()
        if input_shape.ndim > 2:
            raise ValueError(
                "Invalid number of dimensions for input shape array"
                + f" {input_shape.ndim} must be 1 or 2"
            )
        self.input_shape = input_shape[np.newaxis, ...] if input_shape.ndim == 1 else input_shape
        if output_shape.ndim > 2:
            raise ValueError(
                "Invalid number of dimensions for output shape array"
                + f" {output_shape.ndim} must be 2 or 2"
            )
        self.output_shape = (
            output_shape[np.newaxis, ...] if output_shape.ndim == 1 else output_shape
        )

    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> list[torch.Tensor]:
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
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()

    def get_ndims(self) -> int:
        """Get number of dimensions excluding channels dimension.

        Returns
        -------
        int
            number of spatial dimensions.
        """
        return self.input_shape.shape[1] - 1

    def get_input_shape(self, input_id: int = 0) -> NDArray[np.int64]:
        """Get input shape by input index.

        Parameters
        ----------
        input_id : int
            input index.
            Default is 0.

        Returns
        -------
        NDArray[np.int64]
            1 dimensional numpy array
            of int64 representing shape
            of input under given index.

        Raises
        ------
        IndexError
            if no output with given index exists.
        """
        if input_id >= self.input_shape.shape[0]:
            raise IndexError("Invalid index for input")
        return self.input_shape[input_id, :].copy()

    def get_output_shape(self, output_id: int = 0) -> NDArray[np.int64]:
        """Get output shape by output index.

        Parameters
        ----------
        output_id : int
            output index.
            Default is 0.

        Returns
        -------
        NDArray[np.int64]
            1 dimensional numpy array
            of int64 representing shape
            of output under given index.

        Raises
        ------
        IndexError
            if no output with given index exists.
        """
        if output_id >= self.output_shape.shape[0]:
            raise IndexError("Invalid index of output")
        return self.output_shape[output_id, :].copy()

    def get_num_inputs(self) -> int:
        """Get number of input tensors.

        Returns
        -------
        int
            number of input tensors.
        """
        return self.input_shape.shape[0]

    def get_num_outputs(self) -> int:
        """Get number of output tensors.

        Returns
        -------
        int
            number of output tensors.
        """
        return self.output_shape[0]

    @classmethod
    def merge(
        cls, x: torch.Tensor, y: torch.Tensor, how: Literal["+", "*", "."] = "+"
    ) -> torch.Tensor:
        """Merge tensors according given rule.

        Parameters
        ----------
        x : Tensor
            first tensor.
        y : Tensor
            second tensor.
        how : str
            how to merge input tensors. Can be on of following values:
            '+' for sum, '*' for product or '.' for concatenation along first
            dimension. Default is '+'.

        Returns
        -------
        Tensor
            result of merging operation.

        Raises
        ------
        ValueError
            if argument 'how' has value diverging from '+', '*' or '.'.
        """
        if how not in ("+", "*", "."):
            raise ValueError(
                "Argument 'how' must be one of "
                + "following values: ('+', '.', '*'). "
                + "Got {}.".format(how)
            )
        return merge([x, y], how)

    @classmethod
    def crop_as(cls, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Crop first tensor to have the same shape as the second.

        This method affects only spatial dimensions of tensor, so
        batch size and channels dimension remain unchanged.

        Parameters
        ----------
        x : Tensor
            tensor to crop.
        y : Tensor
            tensor whose shape will be used for cropping.

        Returns
        -------
        Tensor
            first tensor cropped to match shape of the second tensor.
        """
        return crop_as(x, y)

    @classmethod
    def pad_as(cls, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Add padding to first tensor to have the same shape as the second.

        This method affects only spatial dimensions of tensor, so
        batch size and channels dimension remain unchanged.


        Parameters
        ----------
        x : Tensor
            tensor to pad.
        y : Tensor
            tensor whose shape will be used for padding size computation.

        Returns
        -------
        Tensor
            first tensor padded to match shape of the second tensor.
        """
        return pad_as(x, y)
