from typing import Literal

import torch


class GatedResidualBlock(torch.nn.Module):
    def __init__(self, pool_method: Literal['none', 'avg', 'max']='none', pool_output_size:int=None):
        super(GatedResidualBlock, self).__init__()
        self.outputs_weight = torch.nn.Parameter(torch.randn(1))

        if pool_method == 'none':
            self.pool = None
        elif pool_method == 'max':
            self.pool = torch.nn.AdaptiveMaxPool1d(output_size=pool_output_size)
        elif pool_method == 'avg':
            self.pool = torch.nn.AdaptiveAvgPool1d(output_size=pool_output_size)        
    
    def forward(self, inputs, outputs): # ((batch_size, max_length, in_channels), (batch_size, max_length, out_channels)) 
                                        # -> (batch_size, max_length, */in_channels+out_channels)
        outputs = torch.sigmoid(self.outputs_weight * outputs) * outputs
        outputs = torch.cat((inputs, outputs), dim=2)

        if self.pool:
            outputs = self.pool(outputs)

        return outputs
