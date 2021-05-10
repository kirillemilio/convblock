""" Contains implementation of various types of inception blocks. """

import os
import sys
from abc import abstractmethod
import math
import numpy as np

from typing import Tuple, List, Dict, Union, Any, Optional
import torch
from torch.nn import functional as F
from ..bases import Sequential, Module
from ..bases import Module, MetaModule, Sequential
from ..layers.conv_block import ResConvBlock as ConvBlock


class InceptionA(Module):

    def __init__(self,
                 input_shape: 'ArrayLike[int]',
                 filters1x1: int = 32,
                 filters3x3dbl: Tuple[int, int, int] = (64, 96, 96),
                 filters5x5: Tuple[int, int] = (48, 64),
                 filters_pool: int = 32,
                 layout: str = 'cna',
                 pool_size: int = 3,
                 kernel_size: int = 5,
                 pool_mode: str = 'avg'):
        super().__init__(input_shape)
        filters1x1 = int(filters1x1)
        filters_pool = int(filters_pool)
        filters3x3dbl = list(map(int, filters3x3dbl))
        filters5x5 = list(map(int, filters5x5))

        pool_size = int(pool_size)

        assert len(filters3x3dbl) == 3
        assert len(filters5x5) == 2
        # assert (self.input_shape[1] == self.input_shape[2]
        #         and self.input_shape[1] == 35)

        self.branch1x1 = ConvBlock(
            input_shape=self.input_shape,
            layout=layout, c=dict(kernel_size=(1, 1),
                                  filters=filters1x1)
        )
        self.branch_pool = ConvBlock(
            input_shape=self.input_shape,
            layout='p ' + layout,
            p=dict(kernel_size=pool_size,
                   stride=1, mode=pool_mode),
            c=dict(kernel_size=(1, 1),
                   filters=filters_pool)
        )
        self.branch5x5 = ConvBlock(
            input_shape=self.input_shape,
            layout=layout * 2,
            c=dict(kernel_size=[(1, 1),
                                (kernel_size,
                                 kernel_size)],
                   filters=filters5x5)
        )
        self.branch3x3dbl = ConvBlock(
            input_shape=self.input_shape,
            layout=layout * 3,
            c=dict(kernel_size=[(1, 1),
                                (3, 3),
                                (3, 3)],
                   filters=filters3x3dbl)
        )

    @property
    def output_shape(self) -> 'ArrayLike[int]':
        total_filters = sum([
            self.branch1x1.out_channels,
            self.branch3x3dbl.out_channels,
            self.branch5x5.out_channels,
            self.branch_pool.out_channels
        ])
        return np.array([
            total_filters,
            *self.branch1x1.output_shape[1:]],
            dtype=np.int
        )

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        return torch.cat([
            self.branch1x1(inputs),
            self.branch3x3dbl(inputs),
            self.branch5x5(inputs),
            self.branch_pool(inputs)
        ], dim=1)


class InceptionB(Module):  # InceptionC in torchvision version

    def __init__(self,
                 input_shape: 'ArrayLike[int]',
                 filters1x1: int = 192,
                 filters7x7: Tuple[int, ...] = (128, 128, 192),
                 filters7x7dbl: Tuple[int, ...] = (128, 128, 128, 128, 192),
                 filters_pool: int = 192,
                 layout: str = 'cna',
                 pool_size: int = 3,
                 pool_mode: str = 'avg'):
        super().__init__(input_shape)
        filters1x1 = int(filters1x1)
        filters_pool = int(filters_pool)
        filters7x7 = list(map(int, filters7x7))
        filters7x7dbl = list(map(int, filters7x7dbl))

        pool_size = int(pool_size)

        assert len(filters7x7) == 3
        assert len(filters7x7dbl) == 5

        # assert (self.input_shape[1] == self.input_shape[2]
        #         and self.input_shape[1] == 17)

        self.branch1x1 = ConvBlock(
            input_shape=self.input_shape,
            layout=layout, c=dict(kernel_size=(1, 1),
                                  filters=filters1x1)
        )
        self.branch_pool = ConvBlock(
            input_shape=self.input_shape, layout='p ' + layout,
            c=dict(kernel_size=(1, 1), filters=filters_pool),
            p=dict(mode=pool_mode, kernel_size=pool_size, stride=1)
        )
        self.branch7x7 = ConvBlock(
            input_shape=self.input_shape,
            layout=layout * 3,
            c=dict(kernel_size=[(1, 1), (1, 7), (7, 1)],
                   filters=filters7x7)
        )
        self.branch7x7dbl = ConvBlock(
            input_shape=self.input_shape,
            layout=layout * 5,
            c=dict(kernel_size=[(1, 1), (7, 1),
                                (1, 7), (7, 1), (1, 7)],
                   filters=filters7x7dbl)
        )

    @property
    def output_shape(self) -> 'ArrayLike[int]':
        total_filters = sum([
            self.branch1x1.out_channels,
            self.branch7x7.out_channels,
            self.branch7x7dbl.out_channels,
            self.branch_pool.out_channels
        ])
        return np.array([
            total_filters,
            *self.branch1x1.output_shape[1:]],
            dtype=np.int
        )

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        return torch.cat([
            self.branch1x1(inputs),
            self.branch7x7(inputs),
            self.branch7x7dbl(inputs),
            self.branch_pool(inputs)
        ], dim=1)


class InceptionC(Module):  # InceptionE in torchvision version

    def __init__(self,
                 input_shape: 'ArrayLike[int]',
                 filters1x1: int = 320,
                 filters3x3: Tuple[int, ...] = (384, 384, 384),
                 filters3x3dbl: Tuple[int, ...] = (448, 384, 384, 384),
                 filters_pool: int = 192,
                 layout: str = 'cna',
                 pool_size: int = 3,
                 pool_mode: str = 'avg',
                 version: int = 3):
        super().__init__(input_shape)

        filters1x1 = int(filters1x1)
        filters_pool = int(filters_pool)
        filters3x3 = list(map(int, filters3x3))
        filters3x3dbl = list(map(int, filters3x3dbl))

        pool_size = int(pool_size)

        assert len(filters3x3) == 3

        if version == 3:
            assert len(filters3x3dbl) == 4
        elif version == 4:
            assert len(filters3x3dbl) == 5
        else:
            raise AssertionError

        # assert (self.input_shape[1] == self.input_shape[2]
        #         and self.input_shape[1] == 8)

        self.branch1x1 = ConvBlock(
            input_shape=self.input_shape,
            layout=layout, c=dict(kernel_size=(1, 1),
                                  filters=filters1x1)
        )

        self.branch_pool = ConvBlock(
            input_shape=self.input_shape,
            layout='p ' + layout, c=dict(kernel_size=(1, 1),
                                         filters=filters_pool),
            p=dict(mode=pool_mode, kernel_size=pool_size, stride=1)
        )

        self.branch3x3 = ConvBlock(
            input_shape=self.input_shape,
            layout=layout, c=dict(kernel_size=(1, 1),
                                  filters=filters3x3[0])
        )
        self.branch3x3_a = ConvBlock(
            input_shape=self.branch3x3.output_shape,
            layout=layout, c=dict(kernel_size=(1, 3),
                                  filters=filters3x3[1])
        )
        self.branch3x3_b = ConvBlock(
            input_shape=self.branch3x3.output_shape,
            layout=layout, c=dict(kernel_size=(3, 1),
                                  filters=filters3x3[2])
        )

        if version == 3:
            self.branch3x3dbl = ConvBlock(
                input_shape=self.input_shape, layout=layout * 2,
                c=dict(kernel_size=[(1, 1), (3, 3)],
                       filters=filters3x3dbl[:2])
            )
        else:
            self.branch3x3dbl = ConvBlock(
                input_shape=self.input_shape, layout=layout * 3,
                c=dict(kernel_size=[(1, 1), (1, 3), (3, 1)],
                       filters=filters3x3dbl[:3])
            )

        self.branch3x3dbl_a = ConvBlock(
            input_shape=self.branch3x3dbl.output_shape,
            layout=layout, c=dict(kernel_size=(1, 3),
                                  filters=filters3x3dbl[-2])
        )
        self.branch3x3dbl_b = ConvBlock(
            input_shape=self.branch3x3dbl.output_shape,
            layout=layout, c=dict(kernel_size=(3, 1),
                                  filters=filters3x3dbl[-1])
        )
        
    @property
    def output_shape(self) -> 'ArrayLike[int]':
        total_filters = sum([
            self.branch1x1.out_channels,
            self.branch3x3_a.out_channels,
            self.branch3x3_b.out_channels,
            self.branch3x3dbl_a.out_channels,
            self.branch3x3dbl_b.out_channels,
            self.branch_pool.out_channels
        ])
        return np.array([
            total_filters,
            *self.branch1x1.output_shape[1:]],
            dtype=np.int
        )

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        outputs3x3 = self.branch3x3(inputs)
        outputs3x3dbl = self.branch3x3dbl(inputs)
        return torch.cat([
            self.branch1x1(inputs),
            self.branch3x3_a(outputs3x3),
            self.branch3x3_b(outputs3x3),
            self.branch3x3dbl_a(outputs3x3dbl),
            self.branch3x3dbl_b(outputs3x3dbl),
            self.branch_pool(inputs)
        ], dim=1)


class ReductionA(Module):  # InceptionB in torchvision version

    def __init__(self,
                 input_shape: 'ArrayLike[int]',
                 filters3x3: int = 384,
                 filters3x3dbl: Tuple[int, int, int] = (64, 96, 96),
                 layout: str = 'cna',
                 pool_size: int = 3,
                 pool_mode: str = 'max'):
        super().__init__(input_shape)

        self.branch3x3 = ConvBlock(
            input_shape=self.input_shape, layout=layout,
            c=dict(kernel_size=3, filters=filters3x3, stride=2)
        )

        self.branch_pool = ConvBlock(
            input_shape=self.input_shape, layout='p',
            p=dict(kernel_size=pool_size,
                   mode=pool_mode, stride=2)
        )

        self.branch3x3dbl = ConvBlock(
            input_shape=self.input_shape,
            layout=layout * 3,
            c=dict(kernel_size=[1, 3, 3],
                   stride=[1, 1, 2],
                   filters=filters3x3dbl)
        )

    @property
    def output_shape(self) -> 'ArrayLike[int]':
        total_filters = sum([
            self.branch3x3.out_channels,
            self.branch3x3dbl.out_channels,
            self.branch_pool.out_channels
        ])
        return np.array([
            total_filters,
            *self.branch3x3.output_shape[1:]],
            dtype=np.int
        )

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        return torch.cat([
            self.branch3x3(inputs),
            self.branch3x3dbl(inputs),
            self.branch_pool(inputs)
        ], dim=1)


class ReductionB(Module):  # InceptionD in torchvision version

    def __init__(self,
                 input_shape: 'ArrayLike[int]',
                 filters3x3: Tuple[int, int] = (192, 320),
                 filters7x7x3: Tuple[int, ...] = (192, 192, 192, 192),
                 layout: str = 'cna',
                 pool_size: int = 3,
                 pool_mode: str = 'avg'):
        super().__init__(input_shape)

        self.branch_pool = ConvBlock(
            input_shape=self.input_shape, layout='p',
            p=dict(kernel_size=pool_size, stride=2, mode=pool_mode)
        )

        self.branch3x3 = ConvBlock(
            input_shape=self.input_shape, layout=layout * 2,
            c=dict(kernel_size=[(1, 1), (3, 3)],
                   filters=filters3x3,
                   stride=[1, 2])
        )

        self.branch7x7x3 = ConvBlock(
            input_shape=self.input_shape, layout=layout * 4,
            c=dict(kernel_size=[(1, 1), (1, 7),
                                (7, 1), (3, 3)],
                   filters=filters7x7x3,
                   stride=[1, 1, 1, 2])
        )

    @property
    def output_shape(self) -> 'ArrayLike[int]':
        total_filters = sum([
            self.branch3x3.out_channels,
            self.branch7x7x3.out_channels,
            self.branch_pool.out_channels
        ])
        return np.array([
            total_filters,
            *self.branch3x3.output_shape[1:]],
            dtype=np.int
        )

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        return torch.cat([
            self.branch3x3(inputs),
            self.branch7x7x3(inputs),
            self.branch_pool(inputs)
        ], dim=1)
