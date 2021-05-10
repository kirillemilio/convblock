""" Contains implementation of SE block. """

import numpy as np
import torch
from torch.nn import functional as F
from ..layers import ConvBlock
from ..bases import Module, MetaModule, Sequential


class SEModule(Module, metaclass=MetaModule):
    
    def __init__(self,
                 input_shape: 'ArrayLike[int]',
                 reduction: int = 4,
                 use_bn: bool = False,
                 use_bias: bool = False,
                 activation: str = 'sigmoid'):
        super().__init__(input_shape)
        filters = self.input_shape[0]
        self.fc = ConvBlock(
            input_shape=self.input_shape,
            layout=('f'
                    + 'n' * use_bn
                    + 'a' + 'f'
                    + 'n' * use_bn
                    + 'a'),
            f=dict(out_features=[filters // reduction,
                                 filters],
                   bias=use_bias),
            a=dict(activation=['relu', activation])
        )
    
    def forward(self, x):
        b, c, *_ = x.size()
        y = F.adaptive_avg_pool2d(x, output_size=1).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1)
        
