"""Contains imports of various custom error."""

from .base_error import BaseError
from .invalid_dims_error import InvalidDimsError
from .invalid_shape_error import InvalidShapeError

__all__ = ["BaseError", "InvalidDimsError", "InvalidShapeError"]
