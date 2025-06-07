"""Contains implementation of lp pooling layer."""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np
import torch

from ...utils import ArrayLike
from .base_pooling_layer import BasePoolLayer


class LPPool(BasePoolLayer):
    """LP-pooling layer implementation generalized for different dimensions.

    Attributes
    ----------
    norm_type : float
        normalization type value for lp-pooling.
    _pool_mode : ClassVar[str]
        pooling mode. "lp" for lp pooling.
    """

    _pool_mode: ClassVar[str] = "lp"
    norm_type: float

    def __init__(
        self,
        input_shape: ArrayLike[int],
        kernel_size: ArrayLike[int] | int,
        stride: ArrayLike[int] | int,
        dilation: ArrayLike[int] | int,
        pad_mode: Literal["constant", "reflect", "replicate", "valid"] = "constant",
        pad_value: float = 0.0,
        norm_type: float = 1.0,
    ):
        """Construct LP-Pooling layer generalized for different dimensions.

        All pooling layers from this module slightly
        extends functionality of original torch.nn.*Pool modules into four
        main aspects:
        1) Shape of the input tensor is passed as argument of constructor.

        2) Shape of the output tensor can be accessed by 'output_shape'
        property of module.

        3) Different padding modes for 'same' mode. It means that there is
        no need to compute padding size for operation to make output tensor
        shape match input tensor's shape. For transposed operations
        'crop' argument can be set to 'True' or 'False'.

        4) All arguments of *Pool operations can be lists, tuples or
        ndarrays of int, np.int, np.int32 or int64 type.

        Parameters
        ----------
        input_shape : ArrayLike[int]
            shape of the input tensor. Note that
            batch dimension is not taken in account.
        kernel_size : ArrayLike[int] | int
            size of pooling kernel along each dimension.
        stride : ArrayLike[int] | int
            size of stride along each dimension. Default is 1.
        dilation : ArrayLike[int] | int
            dilation rate along each dimension. Default is 1.
        pad_mode : Literal["reflect", "replicate", "valid", "constant"]
            padding mode or padding value.
            In case of "constant" `pad_value` will be used for padding
            In case of "valid" no padding will be added.
            Default is "constant".
        pad_value : float
            padding value in case `pad_mode` is "constant".
            Default is 0.0.
        """
        super().__init__(
            input_shape=input_shape,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            pad_mode=pad_mode,
            pad_value=pad_value,
        )
        if np.any(self.to_int_array(self.dilation, "dilation", self.ndims - 1) != 1):
            raise NotImplementedError(
                "Argument 'dilation' that is not equal"
                + " to 1 is not supported"
                + " by LP-Pooling layer."
            )

        self.norm_type = float(norm_type)

    def forward_pool(self, inputs: torch.Tensor, *others: torch.Tensor) -> torch.Tensor:
        """Forward pass method for generalize LPPooling layer.

        Parameters
        ----------
        inputs : torch.Tensor
            input tensor.

        Returns
        -------
        torch.Tensor
            result of pooling operation.
        """
        if self.ndims == 2:
            return torch.nn.functional.lp_pool1d(
                input=inputs,
                norm_type=self.norm_type,
                kernel_size=self.kernel_size,
                stride=self.stride,
            )
        elif self.ndims == 3:
            return torch.nn.functional.lp_pool2d(
                input=inputs,
                norm_type=self.norm_type,
                kernel_size=self.kernel_size,
                stride=self.stride,
            )
        elif self.ndims == 4:
            return torch.nn.functional.lp_pool3d(
                input=inputs,
                norm_type=self.norm_type,
                kernel_size=self.kernel_size,
                stride=self.stride,
            )

        return inputs
