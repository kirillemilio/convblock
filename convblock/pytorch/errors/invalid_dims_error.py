"""Contains implementation of invalid number of dimensions excpetion class."""

from __future__ import annotations

from .base_error import BaseError


class InvalidDimsError(BaseError):
    """Invalid number of dimension error implementation."""

    def __init__(self, ndims: int, target_ndims: int) -> None:
        """Initialize invalid number fo dimensions exception.

        Parameters
        ----------
        ndims : int
            number of dimensions.
        target_dims : int
            target number of dimensions.
        """
        super().__init__(f"Invalid number of dimensions: {ndims}, should be: {target_ndims}")
