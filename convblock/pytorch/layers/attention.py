"""Contains implementation of attention transformer block."""

from __future__ import annotations

import torch


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, input_dim: int, model_dim: int, num_heads: int, dropout_proba: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dropout_proba = float(dropout_proba)

        self.proj_q = torch.nn.Linear(in_features=input_dim, out_feature=model_dim, bias=True)
        self.proj_k = torch.nn.Linear(in_features=input_dim, out_feature=model_dim, bias=True)
        self.proj_v = torch.nn.Linear(in_features=input_dim, out_feature=model_dim, bias=True)

        self.final_proj = torch.nn.Linear(in_features=model_dim, out_features=model_dim, bias=True)

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x.view(x.size(0), x.size(1), self.num_heads, x.size(2) // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def _reshape_output(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size(0), x.size(2)
        return x.permute(0, 2, 1, 3).view(batch_size, seq_len, -1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, q_seq_len = q.size(0), q.size(1)
        k_seq_len = k.size(1)
        q_proj, k_proj, v_proj = (
            self._reshape_input(self.proj_q(q)),
            self._reshape_input(self.proj_k(k)),
            self._reshape_input(self.proj_v(v)),
        )
        atten = (q_proj @ k_proj) / torch.sqrt(k_proj.size(-1))
        if mask is not None:
            _mask = mask.unsqueeze(1).unsqueeze(-1)
            if mask.ndim == 2:
                _mask = mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            elif mask.ndim == 3:
                _mask = mask.unsqueeze(1).unsqueeze(-1)
            else:
                raise RuntimeError(
                    f"Invalid mask shape: {mask.shape} expected [{batch_size}, {q_seq_len}] or"
                    f"[{batch_size}, {q_seq_len}, {k_seq_len}] shape"
                )
            atten.masked_fill_(_mask, -1e12)
        atten = torch.softmax(atten, dim=-1)
        atten = torch.dropout(atten, p=self.dropout_proba, train=self.training)
        output = self._reshape_output(atten.dot(v_proj))
        return self.final_proj(output)


class MobileVitMultiHeadAttention(torch.nn.Module):

    def __init__(self, input_dim: int, model_dim: int, num_heads: int, dropout_proba: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dropout_proba = float(dropout_proba)

        self.proj_q = torch.nn.Linear(in_features=input_dim, out_feature=model_dim, bias=True)
        self.proj_k = torch.nn.Linear(in_features=input_dim, out_feature=model_dim, bias=True)
        self.proj_v = torch.nn.Linear(in_features=input_dim, out_feature=model_dim, bias=True)

        self.final_proj = torch.nn.Linear(in_features=model_dim, out_features=model_dim, bias=True)

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x.view(x.size(0), x.size(1), x.size(2), self.num_heads, x.size(3) // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, num_pixels_in_patch, num_patches, _ = q.size(0), q.size(1), q.size(2)
        q_proj, k_proj, v_proj = (
            self._reshape_input(self.proj_q(q)),
            self._reshape_input(self.proj_k(k)),
            self._reshape_input(self.proj_v(v)),
        )
        atten = torch.softmax(q_proj @ k_proj.transpose(-1, -2), dim=3)
        atten = atten.div(torch.sqrt(k_proj.size(-1)))
        atten = torch.dropout(atten, p=self.dropout_proba, train=self.training)
        output = (
            atten.dot(v_proj)
            .permute(0, 1, 3, 2, 4)
            .view(batch_size, num_pixels_in_patch, num_patches, self.model_dim)
        )
        return self.final_proj(output)
