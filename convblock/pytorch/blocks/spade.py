import torch
from torch.nn import functional as F
from ..layers.conv_block import ResConvBlock as ConvBlock
from ..bases import Module


class SPADE(Module):

    def __init__(self, input_shape, hidden_filters: int = 128, norm_type: str = 'instance'):
        super().__init__(input_shape)
        assert norm_type in ('instance', 'batch')
        assert self.input_shape.shape[0] == 2
        out_filters = self.input_shape[0, 0]
        seg_filters = self.input_shape[1, 0]
        if norm_type == 'instance':
            self.norm = torch.nn.InstanceNorm2d(out_filters, affine=False)
        else:
            self.norm = torch.nn.BatchNorm2d(out_filters, affine=False)
        self.mlp_shared = ConvBlock(
            input_shape=self.input_shape[1], layout='ca c',
            c=dict(kernel_size=3, stride=1,
                   filters=[hidden_filters, 2 * out_filters]),
            a=dict(activation='relu'),
        )

    @property
    def output_shape(self):
        return self.input_shape[0]

    def forward(self, x, seg_map):
        normalized = self.norm(x)
        seg_map = F.interpolate(seg_map,
                                size=x.size()[2:],
                                mode='nearest')
        gamma, beta = torch.chunk(self.mlp_shared(seg_map), chunks=2, dim=1)
        return normalized * (1 + gamma) + beta
        
