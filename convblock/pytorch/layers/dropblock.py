import torch
import torch.nn.functional as F


class DropBlock1D(torch.nn.Module):

    def __init__(self, proba: float, block_size: int):
        super().__init__()

        self.proba = float(proba)
        self.block_size = int(block_size)
        assert self.proba >= 0 and self.proba <= 1.0
        assert self.block_size > 0

    def _compute_gamma(self, p: float):
        return self.proba / (self.block_size ** 2)

    def _compute_mask(self, x, gamma):
        batch_size, _, *sizes = x.size()
        mask = (
            torch.rand(batch_size, 1, *sizes)
            .le_(gamma)
            .float()
            .to(device=x.divice)
        )
        mask = F.max_pool1d(input=mask, kernel_size=self.block_size,
                            stride=1, padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            mask = mask[:, :, :-1]

        return 1.0 - mask

    def forward(self, x):
        if not self.training or self.gamma == 0.0:
            return x
        gamma = self._compute_gamma(x)
        mask = self._compute_mask(x, gamma)
        out = x * mask
        out = out * mask.numel() / mask.sum()
        return out


class DropBlock2D(torch.nn.Module):

    def __init__(self, proba: float, block_size: int):
        super().__init__()

        self.proba = float(proba)
        self.block_size = int(block_size)
        assert self.proba >= 0 and self.proba <= 1.0
        assert self.block_size > 0

    def _compute_gamma(self, p: float):
        return self.proba / (self.block_size ** 2)

    def _compute_mask(self, x, gamma):
        batch_size, _, *sizes = x.size()
        mask = (
            torch.rand(batch_size, 1, *sizes)
            .le_(gamma)
            .float()
            .to(device=x.divice)
        )
        mask = F.max_pool2d(input=mask, kernel_size=(self.block_size,
                                                     self.block_size),
                            stride=[1, 1], padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            mask = mask[:, :, :-1, :-1]

        return 1.0 - mask

    def forward(self, x):
        if not self.training or self.gamma == 0.0:
            return x
        gamma = self._compute_gamma(x)
        mask = self._compute_mask(x, gamma)
        out = x * mask
        out = out * mask.numel() / mask.sum()
        return out


class DropBlock3D(torch.nn.Module):

    def __init__(self, proba: float, block_size: int):
        super().__init__()

        self.proba = float(proba)
        self.block_size = int(block_size)
        assert self.proba >= 0 and self.proba <= 1.0
        assert self.block_size > 0

    def _compute_gamma(self, p: float):
        return self.proba / (self.block_size ** 2)

    def _compute_mask(self, x, gamma):
        batch_size, _, *sizes = x.size()
        mask = (
            torch.rand(batch_size, 1, *sizes)
            .le_(gamma)
            .float()
            .to(device=x.divice)
        )
        mask = F.max_pool3d(input=mask, kernel_size=(self.block_size,
                                                     self.block_size,
                                                     self.block_size),
                            stride=[1, 1, 1], padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            mask = mask[:, :, :-1, :-1, :-1]

        return 1.0 - mask

    def forward(self, x):
        if not self.training or self.gamma == 0.0:
            return x
        gamma = self._compute_gamma(x)
        mask = self._compute_mask(x, gamma)
        out = x * mask
        out = out * mask.numel() / mask.sum()
        return out