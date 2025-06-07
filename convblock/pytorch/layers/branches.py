"""Contains implementation of branches module."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .base_module import BaseModule


class Branches(BaseModule):
    mode: Literal[".", "*", "+", "~"]

    def __init__(self, braches: list[BaseModule], mode: Literal[".", "*", "+", "~"] = ".") -> None:
        if mode not in (".", "+", "*", "~"):
            raise ValueError(
                "Invalid mode for branches."
                + " Must be one of ('+', '.', '*', '~')."
                + " Got '{}' instead.".format(mode)
            )
