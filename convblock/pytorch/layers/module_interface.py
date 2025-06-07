"""Contains implementation of module interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from numpy.typing import NDArray


class IModule(ABC):
    """Module interface implementation."""

    @abstractmethod
    def get_ndims(self) -> int:
        """Get number of dimensions excluding channels dimension.

        Returns
        -------
        int
            number of spatial dimensions.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_num_inputs(self) -> int:
        """Get number of input tensors.

        Returns
        -------
        int
            number of input tensors
        """
        raise NotImplementedError()

    @abstractmethod
    def get_num_outputs(self) -> int:
        """Get number of output tensors.

        Returns
        -------
        int
            number of output tensors.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_input_shape(self, input_idx: int = 0) -> NDArray[np.int64]:
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
            if not input with given index exists.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_output_shape(self, output_idx: int = 0) -> NDArray[np.int64]:
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
        raise NotImplementedError()

    @abstractmethod
    def count_parameters(self, include_static: bool = False) -> int:
        """Count number of parameters.

        Parameters
        ----------
        include_static : bool
            whether to include static(non trainable parameters).
            Default is False meaning that only parameters
            requiring grad will be taken into consideration.

        Returns
        -------
        int
            number of parameters.
        """
        raise NotImplementedError()

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
