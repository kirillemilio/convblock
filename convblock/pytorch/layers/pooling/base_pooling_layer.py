"""Contains pytorch pooling modules compatible with ConvBlock interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

import numpy as np
import torch
from numpy.typing import NDArray

from ...bases import ConvModule
from ...utils import (
    ArrayLike,
    compute_direct_output_shape,
    compute_direct_same_padding,
    pad,
    transform_to_int_tuple,
)


class BasePoolLayer(ABC, ConvModule):
    """Base pooling layer implementation.

    Attributes
    ----------
    _pool_mode : ClassVar[str]
        pooling mode.
    _pad_mode : Literal["constant", "reflect", "replicate", "valid"]
        padding model.
    _pad_value : float
        padding value for constant model.
    _pad_sizes : list[int]
        padding sizes.
    _output_shape : NDArray[np.int32]
        output shape.
    """

    _pool_mode: ClassVar[str] = "base"
    _pad_mode: Literal["constant", "reflect", "replicate", "valid"]
    _pad_value: float
    _pad_sizes: list[int]
    _output_shape: NDArray[np.int32]

    def __init__(
        self,
        input_shape: ArrayLike[int],
        kernel_size: ArrayLike[int] | int,
        stride: ArrayLike[int] | int,
        dilation: ArrayLike[int] | int,
        pad_mode: Literal["reflect", "replicate", "valid", "constant"] = "constant",
        pad_value: float = 0.0,
    ):
        """Construct base class for pooling layers generalized for different dims.

        All pooling layers from this module slightly
        extends functionality of original torch.nn.*Pool modules into four
        main aspects:
        1) Shape of the input tensor is passed as argument of constructor.

        2) Shape of the output tensor can be accessed by 'output_shape'
        property of module.

        3) Different padding modes for 'same' mode. It means that there is
        no need to compute padding size for operation to make output tensor
        shape match input tensor's shape.

        4) All arguments of *Pool operations can be lists, tuples or
        ndarrays of int, np.int, np.int32 or int64 type.

        Parameters
        ----------
        input_shape : ArrayLike[int]
            shape of the input tensor. Note that
            batch dimension is not taken in account.
        kernel_size : ArrayLike[int] | int
            size of deconvolution kernel along each dimension.
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
        ndims = len(input_shape) - 1
        kernel_size_norm = transform_to_int_tuple(kernel_size, "kernel_size", ndims - 1)
        stride_norm = transform_to_int_tuple(stride, "stride", ndims - 1)
        dilation_norm = transform_to_int_tuple(dilation, "dilation", ndims - 1)

        self._pad_mode = pad_mode
        self._pad_value = pad_value
        if self._pad_mode == "valid":
            self._pad_sizes = [0] * (ndims - 1) * 2
        else:
            self._pad_sizes = compute_direct_same_padding(
                kernel_size=kernel_size_norm, stride=stride_norm, dilation=dilation_norm
            )

        _shape = compute_direct_output_shape(
            input_shape=input_shape[1:],
            kernel_size=kernel_size_norm,
            stride=stride_norm,
            dilation=dilation_norm,
            padding=self._pad_sizes,
        )

        output_shape = np.array([input_shape[0], *_shape], dtype=np.int64)
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            kernel_size=kernel_size_norm,
            stride=stride_norm,
            dilation=dilation_norm,
        )

    @property
    def pad_sizes(self) -> list[int]:
        """Get padding sizes.

        Returns
        -------
        list[int]
            list padding sizes.
            [pad_x_l, pad_x_r, ...].
        """
        return self._pad_sizes

    @property
    def pad_mode(self) -> Literal["constant", "reflect", "replicate", "valid"]:
        """Get padding model.

        Returns
        -------
        Literal["constant", "reflect", "replicate", "valid"]
            padding mode.
        """
        return self._pad_mode

    @property
    def pad_value(self) -> float:
        """Get padding value.

        Returns
        -------
        float
            padding value.
        """
        return self._pad_value

    @abstractmethod
    def forward_pool(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pooling abstract method.

        Parameters
        ----------
        inputs : torch.Tensor
            input tensor for pooling.

        Returns
        -------
        torch.Tensor
            output tensor for pooling.
        """
        raise NotImplementedError()

    def forward(self, inputs: torch.Tensor, *others: torch.Tensor) -> torch.Tensor:
        """Forward pass method.

        Parameters
        ----------
        inputs : torch.Tensor

        Returns
        -------
        torch.Tensor
            result of pooling operation.
        """
        x = pad(inputs, self.pad_sizes, mode=self._pad_mode, value=self._pad_value)

        return self.forward_pool(x)

    def __repr__(self) -> str:
        """Get string representation of the module.

        Returns
        -------
        str
            string representation of pooling layer.
        """
        s = "{name}(kernel_size={kernel_size}, stride={stride}"
        if tuple(self.pad_sizes) != (0,) * len(self.pad_sizes):
            s += ", padding={padding}"
        if hasattr(self, "norm_type"):
            s += ", norm_type={norm_type}"
        s += ", mode='{mode}'"
        s += ")"
        values_dict = {
            "kernel_size": self.kernel_size,
            "padding": tuple(self.pad_sizes),
            "stride": self.stride,
            "mode": self.pad_mode,
        }
        return s.format(name=self.__class__.__name__, **values_dict)
