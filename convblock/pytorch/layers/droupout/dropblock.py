"""Contains implementation of drop block technique for n dimensional tensors."""

import torch
import torch.nn.functional as F  # noqa: N812


class DropBlock1D(torch.nn.Module):
    """
    Apply DropBlock regularization to 1D feature maps during training.

    DropBlock randomly drops contiguous blocks of activations, which helps
    regularize models by preventing co-adaptation of nearby units.

    Parameters
    ----------
    proba : float
        Probability of dropping a block.
    block_size : int
        Size of the block to drop.
    """

    proba: float
    block_size: int

    def __init__(self, proba: float, block_size: int):
        super().__init__()
        assert 0.0 <= proba <= 1.0
        assert block_size > 0
        self.proba = float(proba)
        self.block_size = int(block_size)

    def _compute_gamma(self, x: torch.Tensor) -> float:
        length = x.shape[-1]
        return self.proba * length / (self.block_size * (length - self.block_size + 1))

    def _compute_mask(self, x: torch.Tensor, gamma: float) -> torch.Tensor:
        batch_size, _, length = x.size()
        mask = (torch.rand(batch_size, 1, length, device=x.device) < gamma).float()
        mask = F.max_pool1d(
            input=mask,
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2,
        )
        if self.block_size % 2 == 0:
            mask = mask[:, :, :-1]
        return 1.0 - mask

    def __repr__(self) -> str:
        """Get string representation of drop block.

        Returns
        -------
        str
            string representation of dropblock 1d block.
        """
        return f"{self.__class__.__name__}(proba={self.proba}, block_size={self.block_size})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply DropBlock during training and return masked tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, L).

        Returns
        -------
        torch.Tensor
            Tensor with blocks dropped during training.
        """
        if not self.training:
            return x
        gamma = self._compute_gamma(x)
        if gamma == 0.0:
            return x
        mask = self._compute_mask(x, gamma)
        out = x * mask
        out = out * mask.numel() / mask.sum()
        return out


class DropBlock2D(torch.nn.Module):
    """
    Apply DropBlock regularization to 2D feature maps during training.

    DropBlock helps reduce overfitting by dropping spatially contiguous
    regions from feature maps.

    Parameters
    ----------
    proba : float
        Probability of dropping a block.
    block_size : int
        Size of the square block to drop.
    """

    proba: float
    block_size: int

    def __init__(self, proba: float, block_size: int):
        super().__init__()
        assert 0.0 <= proba <= 1.0
        assert block_size > 0
        self.proba = float(proba)
        self.block_size = int(block_size)

    def _compute_gamma(self, x: torch.Tensor) -> float:
        h, w = x.shape[-2:]
        return (
            self.proba
            * h
            * w
            / (self.block_size**2 * (h - self.block_size + 1) * (w - self.block_size + 1))
        )

    def _compute_mask(self, x: torch.Tensor, gamma: float) -> torch.Tensor:
        batch_size, _, h, w = x.size()
        mask = (torch.rand(batch_size, 1, h, w, device=x.device) < gamma).float()
        mask = F.max_pool2d(
            input=mask,
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2,
        )
        if self.block_size % 2 == 0:
            mask = mask[:, :, :-1, :-1]
        return 1.0 - mask

    def __repr__(self) -> str:
        """Get string representation of drop block.

        Returns
        -------
        str
            string representation of dropblock 1d block.
        """
        return f"{self.__class__.__name__}(proba={self.proba}, block_size={self.block_size})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply DropBlock during training and return masked tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Tensor with blocks dropped during training.
        """
        if not self.training:
            return x
        gamma = self._compute_gamma(x)
        if gamma == 0.0:
            return x
        mask = self._compute_mask(x, gamma)
        out = x * mask
        out = out * mask.numel() / mask.sum()
        return out


class DropBlock3D(torch.nn.Module):
    """
    Apply DropBlock regularization to 3D feature maps during training.

    DropBlock drops 3D blocks of activations to improve generalization.

    Parameters
    ----------
    proba : float
        Probability of dropping a block.
    block_size : int
        Size of the cubic block to drop.
    """

    proba: float
    block_size: int

    def __init__(self, proba: float, block_size: int):
        super().__init__()
        assert 0.0 <= proba <= 1.0
        assert block_size > 0
        self.proba = float(proba)
        self.block_size = int(block_size)

    def _compute_gamma(self, x: torch.Tensor) -> float:
        d, h, w = x.shape[-3:]
        return (
            self.proba
            * d
            * h
            * w
            / (
                self.block_size**3
                * (d - self.block_size + 1)
                * (h - self.block_size + 1)
                * (w - self.block_size + 1)
            )
        )

    def _compute_mask(self, x: torch.Tensor, gamma: float) -> torch.Tensor:
        batch_size, _, d, h, w = x.size()
        mask = (torch.rand(batch_size, 1, d, h, w, device=x.device) < gamma).float()
        mask = F.max_pool3d(
            input=mask,
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2,
        )
        if self.block_size % 2 == 0:
            mask = mask[:, :, :-1, :-1, :-1]
        return 1.0 - mask

    def __repr__(self) -> str:
        """Get string representation of drop block.

        Returns
        -------
        str
            string representation of dropblock 1d block.
        """
        return f"{self.__class__.__name__}(proba={self.proba}, block_size={self.block_size})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply DropBlock during training and return masked tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, D, H, W).

        Returns
        -------
        torch.Tensor
            Tensor with blocks dropped during training.
        """
        if not self.training:
            return x
        gamma = self._compute_gamma(x)
        if gamma == 0.0:
            return x
        mask = self._compute_mask(x, gamma)
        out = x * mask
        out = out * mask.numel() / mask.sum()
        return out
