"""Contains implementation of attention factory class."""

from __future__ import annotations

from typing import Literal

from ...initializer import InitializerConfigDict
from ...utils import ArrayLike
from ..torch_module import TorchModule
from .mvit_patch_attention import MVitPatchMultiHeadAttentionModule
from .swin_window_attention import SwinWindowAttentionModule
from .vit_multihead_attention import VitMultiHeadAttentionModule


class AttentionFactory:
    """
    Factory for constructing various attention modules.

    Supports multiple attention mechanisms used in vision models,
    including ViT-style full attention, MobileViT patch-based attention,
    and Swin window-based attention.

    Methods
    -------
    create_attention(...)
        Create and return an attention module based on selected mode.
    """

    @classmethod
    def create_attention(
        cls,
        input_shape: ArrayLike[int],
        num_heads: int,
        mode: Literal[
            "vit-mha",
            "mvit-patch",
            "swin-window",
            "axial",
            "coatnet",
            "cswin",
            "halo",
            "maxvit",
        ] = "vit-mha",
        patch_size: ArrayLike[int] | int = 1,
        in_filters: int | None = None,
        out_filters: int | None = None,
        dropout_prob: float = 0.1,
        bias: bool = False,
        init_weight: InitializerConfigDict | None = None,
        init_bias: InitializerConfigDict | None = None,
    ) -> TorchModule:
        """
        Create attention module based on selected mode.

        Parameters
        ----------
        input_shape : ArrayLike[int]
            Shape of the input tensor [C, ...spatial...].
        num_heads : int
            Number of attention heads.
        mode : str, default="vit-mha"
            Attention type. One of:
            - "vit-mha"     : Full spatial attention (ViT-style)
            - "mvit-patch"  : Patch-wise local attention (MobileViT)
            - "swin-window" : Shifted window attention (Swin)
            - "axial"       : Axial sequential attention
            - "coatnet"     : Hybrid conv-attention (CoAtNet)
            - "cswin"       : Cross-shaped window attention
            - "halo"        : HaloNet attention
            - "maxvit"      : MaxViT grid+block attention
        patch_size : int or tuple of int
            Patch size for attention (used in local/patched modes).
        in_filters : int, optional
            Input filters to projection layers.
        out_filters : int, optional
            Output filters to projection layers.
        dropout_prob : float, default=0.1
            Dropout rate applied in attention.
        bias : bool, default=False
            Whether to use bias in projection layers.
        init_weight : InitializerConfigDict, optional
            Weight initialization config.
        init_bias : InitializerConfigDict, optional
            Bias initialization config.

        Returns
        -------
        TorchModule
            Initialized attention module.
        """
        if mode == "vit-mha":
            return VitMultiHeadAttentionModule(
                input_shape=input_shape,
                num_heads=num_heads,
                in_filters=in_filters,
                out_filters=out_filters,
                dropout_prob=dropout_prob,
                bias=bias,
                init_weight=init_weight,
                init_bias=init_bias,
            )
        elif mode == "mvit-patch":
            return MVitPatchMultiHeadAttentionModule(
                input_shape=input_shape,
                num_heads=num_heads,
                patch_size=patch_size,
                in_filters=in_filters,
                out_filters=out_filters,
                dropout_prob=dropout_prob,
                bias=bias,
                init_weight=init_weight,
                init_bias=init_bias,
            )
        elif mode == "swin-window":
            return SwinWindowAttentionModule(
                input_shape=input_shape,
                window_size=patch_size,
                num_heads=num_heads,
                in_filters=in_filters,
                out_filters=out_filters,
                dropout_prob=dropout_prob,
                bias=bias,
                init_weight=init_weight,
                init_bias=init_bias,
            )
        elif mode == "axial":
            # Axial attention: sequential attention over individual axes
            # Paper: https://arxiv.org/abs/2003.07853
            raise NotImplementedError("Axial attention is not implemented yet.")
        elif mode == "coatnet":
            # CoAtNet: Conv + Attention hybrid
            # Paper: https://arxiv.org/abs/2106.04803
            raise NotImplementedError("CoAtNet attention block is not implemented yet.")
        elif mode == "cswin":
            # CSWin: Cross-shaped window attention
            # Paper: https://arxiv.org/abs/2107.00652
            raise NotImplementedError("CSWin attention is not implemented yet.")
        elif mode == "halo":
            # HaloNet: local attention with context overlap
            # Paper: https://arxiv.org/abs/2103.12731
            raise NotImplementedError("Halo attention is not implemented yet.")
        elif mode == "maxvit":
            # MaxViT: grid attention + block attention
            # Paper: https://arxiv.org/abs/2204.01697
            raise NotImplementedError("MaxViT attention is not implemented yet.")
        else:
            raise ValueError(f"Unknown attention type: {mode}")
