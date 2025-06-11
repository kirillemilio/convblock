"""Contains implementation of deformable convolutions modulation layer."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noa: N812
from tvdcn import deform_conv1d, deform_conv2d, deform_conv3d

from ...initializer import InitializerConfigDict
from ...utils import ArrayLike
from ..modulated_torch_module import ModulatedTorchModule
