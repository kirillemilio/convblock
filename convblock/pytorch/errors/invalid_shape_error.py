"""Contains implementation of invalid shape exception class."""

from __future__ import annotations

from .base_error import BaseError


class InvalidShapeError(BaseError):
    """Invalid shape error implementation."""

    def __init__(self, shape: tuple[int, ...], target_shape: tuple[int, ...]) -> None:
        """Initialize invalid shape exception.

        Parameters
        ----------
        shape : tuple[int, ...]
            shape of tensor.
        target_shape : tuple[int, ...]
            target shape of tensor.
        """
        super().__init__(f"Invalid shape of tensor: {shape}, " + f"should be: {target_shape}")
