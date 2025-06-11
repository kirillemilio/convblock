"""Contains implementation of modulated torch module."""

from __future__ import annotations

import numpy as np

from ..utils import ArrayLike
from .torch_module import TorchModule


class ModulatedTorchModule(TorchModule):
    """Modulated torch module implementation."""

    def __init__(
        self,
        input_shape: ArrayLike[int],
        modulation_shape: ArrayLike[int],
        output_shape: ArrayLike[int],
    ) -> None:
        super().__init__(
            input_shape=np.array([input_shape, modulation_shape], dtype=np.int64),
            output_shape=output_shape,
        )
