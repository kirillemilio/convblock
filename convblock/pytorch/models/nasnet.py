from ..layers.conv_block import ConvBlock, ConvBranches
from ..config import Config
from .base_model import BaseModel
from ..bases import Module, Sequential
from ..blocks.nas_cell import *

        

class StemBlock(Module):
    
    def __init__(self, input_shape, config):
        super().__init__(input_shape)
        self.init_block = ConvBlock(
            input_shape=input_shape,
            **config.get('init')
        )
        self.first_stem = CellStem0(input_shape=self.init_block.output_shape,
                                    filters=config.get('stem/filters')[0])
        self.second_stem = CellStem1(input_shape=np.stack([self.init_block.output_shape,
                                                           self.first_stem.output_shape], axis=0),
                                     filters=config.get('stem/filters')[1])
                                     
    @property
    def output_shape(self):
        return np.stack([self.first_stem.output_shape,
                         self.second_stem.output_shape], axis=0)
    
    def forward(self, x):
        y = self.init_block(x)
        z = self.first_stem(y)
        return z, self.second_stem([y, z])



class NASNetA(BaseModel):
    
    @classmethod
    def default_config(cls):
        """ Get default config for PeleeNet model. """
        config = BaseModel.default_config()
        config['input'] = {
            'init': {
                'layout': 'cn',
                'c': {
                    'kernel_size': 3,
                    'stride': 2,
                    'filters': 32
                }
            },
            'stem': {
                'filters': [32, 32]
            }
        }

        config['body'] = {
            'cell_filters': [168, 168 * 2, 168 * 4],
            'num_cells': [6, 6, 6],
            'reduction_filters': [168 * 2, 168 * 4, 168 * 8],
            'use_reduction': [True, True, False]
            
        }

        config['head'] = None
        return config
    
    def build_input(self, input_shape, config):
        return StemBlock(input_shape=input_shape, config=config)
    
    def build_body(self, input_shape, config):
        cell_filters = config.get('cell_filters')
        num_cells = config.get('num_cells')
        reduction_filters = config.get('reduction_filters')
        use_reduction = config.get('use_reduction')
        shape = input_shape
        
        all_blocks = []
        for i, (c, n, r, u) in enumerate(zip(cell_filters,
                                             num_cells,
                                             reduction_filters,
                                             use_reduction)):
            for j in range(n):
                if j == 0:
                    block = FirstCell(input_shape=shape, filters=c, use_prev_proj_output=False)
                else:
                    block = NormalCell(input_shape=shape, filters=c)
                shape = block.output_shape
                all_blocks.append(block)
            
            if u:
                block = ReductionCell(input_shape=shape, filters=r)
                shape = block.output_shape
                all_blocks.append(block)
        return Sequential(*all_blocks)