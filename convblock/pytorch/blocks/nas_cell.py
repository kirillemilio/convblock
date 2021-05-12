""" Contains implementation of NasNet architecture convolutional cells. """

import torch
from ..layers import ConvBlock
from ..layers.conv_block import ConvBranches
from ..bases import Module


class NasCell(Module):
    
    @classmethod
    def branch_separables(cls,
                          input_shape,
                          out_channels,
                          kernel_size,
                          downsample=False,
                          mode=0, **kwargs):
        in_channels = input_shape[0]
        if mode == 0:
            return ConvBlock(
                input_shape=input_shape, layout='a ccna ccn',
                c=dict(kernel_size=[kernel_size, 1, kernel_size, 1],
                       filters=[in_channels, in_channels, in_channels, out_channels],
                       stride=[2 if downsample else 1, 1, 1, 1]),
                **kwargs
            )
        elif mode == 1:
            return ConvBlock(
                input_shape=input_shape, layout='a ccna ccn',
                c=dict(kernel_size=[kernel_size, 1, kernel_size, 1],
                       filters=[in_channels, out_channels, out_channels, out_channels],
                       stride=[2 if downsample else 1, 1, 1, 1]),
                **kwargs
            )
        else:
            raise ValueError('Unknown mode')


class CellStem0(NasCell):
    
    def __init__(self, input_shape, filters):
        super().__init__(input_shape=input_shape)

        self.conv1x1 = ConvBlock(
            input_shape=input_shape, layout='nac',
            c=dict(kernel_size=1, stride=1, filters=filters)
        )

        self.branch_left_1 = self.branch_separables(
            input_shape=self.conv1x1.output_shape,
            out_channels=filters, kernel_size=5,
            downsample=True, mode=0
        )
        
        self.branch_right_1 = self.branch_separables(
            input_shape=self.input_shape,
            out_channels=filters, kernel_size=7,
            downsample=True, mode=1
        )
        
        self.branch_left_2 = ConvBlock(
            input_shape=self.conv1x1.output_shape,
            layout='p', p=dict(mode='max', kernel_size=3, stride=2)
        )
        
        self.branch_right_2 = self.branch_separables(
            input_shape=self.input_shape,
            out_channels=filters,
            kernel_size=7,
            downsample=True,
            mode=1
        )
        
        self.branch_left_3 = ConvBlock(
            input_shape=self.conv1x1.output_shape,
            layout='p', p=dict(mode='avg', kernel_size=3, stride=2)
        )
        
        self.branch_right_3 = self.branch_separables(
            input_shape=self.input_shape,
            out_channels=filters,
            kernel_size=5,
            downsample=True,
            mode=1
        )
        
        self.branch_right_4 = ConvBlock(
            input_shape=self.branch_left_1.output_shape,
            layout='p', p=dict(kernel_size=3, stride=1, mode='avg')
        )
        
        self.branch_left_5 = self.branch_separables(
            input_shape=self.branch_left_1.output_shape,
            out_channels=filters,
            kernel_size=3,
            downsample=False,
            mode=0
        )
        
        self.branch_right_5 = ConvBlock(
            input_shape=self.conv1x1.output_shape,
            layout='p', p=dict(kernel_size=3, stride=2, mode='max')
        )
        
    @property
    def output_shape(self):
        return np.array([
            self.branch_right_5.output_shape[0] * 4,
            *self.branch_right_5.output_shape[1:]
        ])
        
    def forward(self, x):
        y = self.conv1x1(x)

        z1 = self.branch_left_1(y) + self.branch_right_1(x)
        z2 = self.branch_left_2(y) + self.branch_right_2(x)
        z3 = self.branch_left_3(y) + self.branch_right_3(x)
        
        z4 = self.branch_right_4(z1) + z2
        
        z5 = self.branch_left_5(z1) + self.branch_right_5(y)
        
        return torch.cat([z2, z3, z4, z5], dim=1)
    

class CellStem1(NasCell):
    

    def __init__(self, input_shape, filters):
        super().__init__(input_shape=input_shape)
        
        self.conv1x1 = ConvBlock(
            input_shape=self.input_shape[1], layout='acn',
            c=dict(kernel_size=1, stride=1, filters=filters)
        )
        
        self.path_right = ConvBlock(
            input_shape=self.input_shape[0], layout='a pcn',
            p=dict(mode='avg', kernel_size=3, stride=2),
            c=dict(kernel_size=1, filters=filters, stride=1)
        )

        self.branch_left_1 = self.branch_separables(
            input_shape=self.conv1x1.output_shape,
            out_channels=filters, kernel_size=5,
            downsample=True, mode=0
        )
        
        self.branch_right_1 = self.branch_separables(
            input_shape=self.path_right.output_shape,
            out_channels=filters, kernel_size=7,
            downsample=True, mode=1
        )
        
        self.branch_left_2 = ConvBlock(
            input_shape=self.conv1x1.output_shape,
            layout='p', p=dict(mode='max', kernel_size=3, stride=2)
        )
        
        self.branch_right_2 = self.branch_separables(
            input_shape=self.path_right.output_shape,
            out_channels=filters,
            kernel_size=7,
            downsample=True,
            mode=1
        )
        
        self.branch_left_3 = ConvBlock(
            input_shape=self.conv1x1.output_shape,
            layout='p', p=dict(mode='avg', kernel_size=3, stride=2)
        )
        
        self.branch_right_3 = self.branch_separables(
            input_shape=self.path_right.output_shape,
            out_channels=filters,
            kernel_size=5,
            downsample=True,
            mode=1
        )
        
        self.branch_right_4 = ConvBlock(
            input_shape=self.branch_left_1.output_shape,
            layout='p', p=dict(kernel_size=3, stride=1, mode='avg')
        )
        
        self.branch_left_5 = self.branch_separables(
            input_shape=self.branch_left_1.output_shape,
            out_channels=filters,
            kernel_size=3,
            downsample=False,
            mode=0
        )
        
        self.branch_right_5 = ConvBlock(
            input_shape=self.conv1x1.output_shape,
            layout='p', p=dict(kernel_size=3, stride=2, mode='max')
        )
        
    @property
    def output_shape(self):
        return np.array([
            self.branch_right_5.output_shape[0] * 4,
            *self.branch_right_5.output_shape[1:]
        ])
        
    def forward(self, inputs):
        x_prev, x = inputs
        x_left = self.conv1x1(x)
        x_right = self.path_right(x_prev)
        
        y1 = self.branch_left_1(x_left) + self.branch_right_1(x_right)
        y2 = self.branch_left_2(x_left) + self.branch_right_2(x_right)
        y3 = self.branch_left_3(x_left) + self.branch_right_3(x_right)
        
        y4 = self.branch_right_4(y1) + y2
        
        y5 = self.branch_left_5(y1) + self.branch_right_5(x_left)
        return torch.cat([y2, y3, y4, y5])


class FirstCell(NasCell):
    
    def __init__(self, input_shape, filters, use_prev_proj_output=False):
        super().__init__(input_shape)
        self.use_prev_proj_output = bool(use_prev_proj_output)
        # h[t-1], h[t]
        self.conv1x1 = ConvBlock(
            input_shape=self.input_shape[1], layout='acn',
            c=dict(kernel_size=1, filters=filters, stride=1)
        )
        
        self.path_left = ConvBlock(
            input_shape=self.input_shape[0], layout='pcn',
            c=dict(kernel_size=1, stride=1, filters=filters, groups=2),
            p=dict(mode='avg', kernel_size=1, stride=2)
        )
        
        self.branch_left_1 = self.branch_separables(
            input_shape=self.conv1x1.output_shape,
            out_channels=filters,
            kernel_size=5,
            downsample=False,
            mode=0
        )
        
        self.branch_right_1 = self.branch_separables(
            input_shape=self.path_left.output_shape,
            out_channels=filters,
            kernel_size=3,
            downsample=False,
            mode=0
        )
        
        self.branch_left_2 = self.branch_separables(
            input_shape=self.path_left.output_shape,
            out_channels=filters,
            kernel_size=5,
            downsample=False,
            mode=0
        )
        
        self.branch_right_2 = self.branch_separables(
            input_shape=self.path_left.output_shape,
            out_channels=filters,
            kernel_size=3,
            downsample=False,
            mode=0
        )
        
        self.branch_left_3 = ConvBlock(
            input_shape=self.conv1x1.output_shape,
            layout='p', p=dict(mode='avg', kernel_size=3, stride=1)
        )
        
        self.branch_left_4 = ConvBlock(
            input_shape=self.path_left.output_shape,
            layout='p', p=dict(mode='avg', kernel_size=3, stride=1)
        )

        self.branch_right_4 = ConvBlock(
            input_shape=self.path_left.output_shape,
            layout='p', p=dict(mode='avg', kernel_size=3, stride=1)
        )
        
        self.branch_right_5 = self.branch_separables(
            input_shape=self.conv1x1.output_shape,
            out_channels=filters,
            kernel_size=3,
            downsample=False,
            mode=0
        )
        
    @property
    def output_shape(self):
        shape = np.array([
            self.branch_right_5.output_shape[0] * (6 if self.use_prev_proj_output else 5),
            *self.branch_right_5.output_shape[1:]
        ])
        return np.stack([self.input_shape[1], shape], axis=0)
        
    def forward(self, inputs):
        x_prev, x = inputs
        x_left, x_right = self.path_left(x_prev), self.conv1x1(x)
        
        y1 = self.branch_left_1(x_right) + self.branch_right_1(x_left)
        y2 = self.branch_left_2(x_left) + self.branch_right_2(x_left)
        
        y_mid = self.branch_left_3(x_right) + x_left
        
        y3 = self.branch_left_4(x_left) + self.branch_right_4(x_left)
        y4 = self.branch_right_5(x_right) + x_right
        
        if self.use_prev_proj_output:
            return x, torch.cat([x_left, y1, y2, y3, y4, y5], dim=1)
        else:
            return x, torch.cat([y1, y2, y3, y4, y5], dim=1)
    

class NormalCell(NasCell):
    
    def __init__(self, input_shape, filters, use_prev_proj_output=False):
        super().__init__(input_shape)
        self.use_prev_proj_output = bool(use_prev_proj_output)
        self.conv1x1 = ConvBlock(
            input_shape=input_shape[1], layout='acn',
            c=dict(kernel_size=1, filters=filters, stride=1)
        )
        
        self.conv1x1_prev = ConvBlock(
            input_shape=input_shape[0], layout='acn',
            c=dict(kernel_size=1, filters=filters, stride=1)
        )
        
        self.branch_left_1 = self.branch_separables(
            input_shape=self.conv1x1.output_shape,
            out_channels=filters,
            kernel_size=5,
            downsample=False,
            mode=0
        )
        self.branch_right_1 = self.branch_separables(
            input_shape=self.conv1x1_prev.output_shape,
            out_channels=filters,
            kernel_size=3,
            downsample=False,
            mode=0
        )
        
        self.branch_left_2 = self.branch_separables(
            input_shape=self.conv1x1_prev.output_shape,
            out_channels=filters,
            kernel_size=5,
            downsample=False,
            mode=0
        )
        self.branch_right_2 = self.branch_separables(
            input_shape=self.conv1x1_prev.output_shape,
            out_channels=filters,
            kernel_size=3,
            downsample=False,
            mode=0
        )
        
        self.branch_left_3 = ConvBlock(
            input_shape=self.conv1x1.output_shape,
            layout='p', p=dict(mode='avg', kernel_size=3, stride=1)
        )
        
        self.branch_left_4 = ConvBlock(
            input_shape=self.conv1x1_prev.output_shape,
            layout='p', p=dict(mode='avg', kernel_size=3, stride=1)
        )
        self.branch_right_4 = ConvBlock(
            input_shape=self.conv1x1_prev.output_shape,
            layout='p', p=dict(mode='avg', kernel_size=3, stride=1)
        )
        
        self.branch_left_5 = self.branch_separables(
            input_shape=self.conv1x1.output_shape,
            out_channels=filters,
            kernel_size=3,
            downsample=False,
            mode=0
        )
        
    @property
    def output_shape(self):
        shape = np.array([
            self.branch_left_5.output_shape[0] * (6 if self.use_prev_proj_output else 5),
            *self.branch_left_5.output_shape[1:]
        ])
        return np.stack([self.input_shape[1], shape], axis=0)
        
    def forward(self, inputs):
        x_prev, x = inputs
        x_left, x_right = self.conv1x1_prev(x_prev), self.conv1x1(x)
        
        y1 = self.branch_left_1(x_right) + self.branch_right_1(x_left)
        y2 = self.branch_left_2(x_left) + self.branch_right_2(x_left)
        y3 = self.branch_left_3(x_right) + x_left
        
        y4 = self.branch_left_4(x_left) + self.branch_right_4(x_left)
        
        y5 = self.branch_left_5(x_right) + x_right
        
        if self.use_prev_proj_output:
            return x, torch.cat([x_left, y1, y2, y3, y4, y5], dim=1)
        else:
            return x, torch.cat([y1, y2, y3, y4, y5], dim=1)


class ReductionCell(NasCell):
    
    def __init__(self, input_shape, filters, use_second_block_output=False):
        super().__init__(input_shape)
        self.use_second_block_output = bool(use_second_block_output)
        self.conv1x1 = ConvBlock(
            input_shape=self.input_shape[1],
            layout='acn', c=dict(filters=filters, kernel_size=1, stride=1)
        )
        
        self.conv1x1_prev = ConvBlock(
            input_shape=self.input_shape[0],
            layout='acn', c=dict(filters=filters, kernnel_size=1, stride=1)
        )
        
        self.branch_left_1 = self.branch_separables(
            input_shape=self.conv1x1.output_shape,
            out_channels=filters,
            kernel_size=5,
            downsample=True,
            mode=0
        )
        self.branch_right_1 = self.branch_separables(
            input_shape=self.conv1x1_prev.output_shape,
            out_channels=filters,
            kernel_size=7,
            downsample=True,
            mode=0
        )
        
        self.branch_left_2 = ConvBlock(
            input_shape=self.conv1x1.output_shape,
            layout='p', p=dict(mode='max', kernel_size=3, stride=2)  
        )
        self.branch_right_2 = self.branch_separables(
            input_shape=self.conv1x1_prev.output_shape,
            out_channels=filters,
            kernel_size=7,
            downsample=True,
            mode=0
        )
        
        self.branch_left_3 = ConvBlock(
            input_shape=self.conv1x1.output_shape,
            layout='p', p=dict(mode='avg', kernel_size=3, stride=2)
        )
        self.branch_right_3 = self.branch_separables(
            input_shape=self.conv1x1_prev.output_shape,
            out_channels=filters,
            kernel_size=5,
            downsample=True,
            mode=0
        )
        
        self.branch_right_4 = ConvBlock(
            input_shape=self.branch_left_1.output_shape,
            layout='p', p=dict(mode='avg', kernel_size=3, stride=1)
        )
        
        self.branch_left_5 = self.branch_separables(
            input_shape=self.branch_left_1.output_shape,
            out_channels=filters,
            kernel_size=3,
            downsample=False,
            mode=0
        )
        self.branch_right_5 = ConvBlock(
            input_shape=self.conv1x1.output_shape,
            layout='p', p=dict(mode='max', kernel_size=3, stride=2)
        )
    
    @property
    def output_shape(self):
        shape = np.array([
            self.branch_right_5.output_shape[0] * (4 if self.use_second_block_output else 3),
            *self.branch_right_5.output_shape[1:]
        ])
        return np.stack([self.input_shape[1], shape], axis=0)
        

    def forward(self, inputs):
        x_prev, x = inputs
        x_left, x_right = self.conv1x1_prev(x_prev), self.conv1x1(x)
        
        y1 = self.branch_left_1(x_right) + self.branch_right_1(x_left)
        y2 = self.branch_left_2(x_right) + self.branch_right_2(x_left)
    
        y3 = self.branch_left_3(x_right) + self.branch_right_3(x_left)
        y4 = self.branch_right_4(y1) + y2

        y5 = self.branch_left_5(y1) + self.branch_right_5(x_right)
        
        if self.use_second_block_output:
            return x, torch.cat([y2, y3, y4, y5], dim=1)
        else:
            return x, torch.cat([y3, y4, y5], dim=1)