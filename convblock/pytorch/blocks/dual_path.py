import math
import numpy as np
import torch
from torch.nn import functional as F
from ..layers import ConvBlock
from ..bases import Module


class DPNBlock(Module):

    def __init__(self,
                 input_shape: 'ArrayLike[int]',
                 filters: 'ArrayLike[int]',
                 inc: int,
                 groups: int = 1,
                 downsample: bool = False,
                 force_proj: bool = False,
                 layout: str = 'nac'):
        super().__init__(input_shape)
        self.downsample = bool(downsample)
        self.force_proj = bool(force_proj)
        if self.num_inputs == 2:
            shape = self.input_shape[0, :]
            shape[0] += self.input_shape[1, 0]
        else:
            shape = self.input_shape[:]
        filters = list(filters)

        self.filters = filters[:]
        self.inc = int(inc)
        self.main = ConvBlock(
            input_shape=shape, layout=layout * 3,
            c=dict(filters=[self.filters[0],
                            self.filters[1],
                            self.filters[2] + inc],
                   groups=[1, groups, 1],
                   stride=(1 if not downsample else (1, 2, 1)),
                   kernel_size=[1, 3, 1])
        )

        if force_proj or downsample:
            self.proj = ConvBlock(
                input_shape=shape, layout=layout,
                c=dict(filters=self.filters[-1] + 2 * self.inc,
                       kernel_size=1,
                       stride=(1 if not downsample else 2))
            )
        else:
            self.proj = None

    @property
    def output_shape(self):
        res_shape = self.main.output_shape.copy()
        res_shape[0] -= self.inc

        if self.proj is not None:
            dense_shape = self.proj.output_shape.copy()
            dense_shape[0] -= self.filters[-1]
            dense_shape[0] += self.inc
        else:
            dense_shape = res_shape.copy()
            dense_shape[0] = self.input_shape[1, 0]
            dense_shape[0] += self.inc
        return np.stack([res_shape, dense_shape], axis=0)

    def forward(self, x):
        x_in = (torch.cat(x, dim=1)
                if isinstance(x, (tuple, list)) else x)
        if self.proj is not None:
            x_s = self.proj(x_in)
            x_s1 = x_s[:, :self.filters[-1], :, :]
            x_s2 = x_s[:, self.filters[-1]:, :, :]
        else:
            x_s1 = x[0]
            x_s2 = x[1]

        x_in = self.main(x_in)
        out1 = x_in[:, :self.filters[-1], :, :]
        out2 = x_in[:, self.filters[-1]:, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense
