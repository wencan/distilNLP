from typing import Literal, Sequence, Union
import warnings

import torch


class GatedResidual(torch.nn.Module):
    def __init__(self):
        super(GatedResidual, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(1))
    
    def forward(self, inputs: torch.Tensor, outputs: torch.tensor): 
                                        # ((batch_size, max_length, in_channels), (batch_size, max_length, out_channels)) 
                                        # -> (batch_size, max_length, pool_output_size or in_channels+out_channels)
        gate = torch.nn.functional.sigmoid(outputs*self.weight)
        outputs = torch.cat((inputs*(1-gate), outputs*gate), dim=-1)

        return outputs


class GatedResidualNet(torch.nn.Module):
    def __init__(self, module:torch.nn.Module, module_name:str, downsample_method: Literal['none', 'avg', 'max']='none', downsample_size:int=None):
        super(GatedResidualNet, self).__init__()

        self.module_name = module_name
        setattr(self, module_name, module)
        self.residual = GatedResidual()

        if downsample_method == 'none':
            self.downsample = None
        elif downsample_method == 'max':
            self.downsample = torch.nn.AdaptiveMaxPool1d(output_size=downsample_size)
        elif downsample_method == 'avg':
            self.downsample = torch.nn.AdaptiveAvgPool1d(output_size=downsample_size)
    
    def forward(self, x):
        x_copy = x.clone()
        out = getattr(self, self.module_name)(x)
        out = self.residual(x_copy, out)

        if self.downsample:
            out = self.downsample(out)

        return out


class GatedResidualBlock(torch.nn.Module):
    def __init__(self, pool_method: Literal['none', 'avg', 'max']='none', pool_output_size:int=None):
        warnings.warn('GatedResidualBlock is deprecated. Please use GatedResidual or GatedResidualNet instead.')

        super(GatedResidualBlock, self).__init__()
        self.outputs_weight = torch.nn.Parameter(torch.randn(1))

        if pool_method == 'none':
            self.pool = None
        elif pool_method == 'max':
            self.pool = torch.nn.AdaptiveMaxPool1d(output_size=pool_output_size)
        elif pool_method == 'avg':
            self.pool = torch.nn.AdaptiveAvgPool1d(output_size=pool_output_size)        
    
    def forward(self, inputs, outputs): # ((batch_size, max_length, in_channels), (batch_size, max_length, out_channels)) 
                                        # -> (batch_size, max_length, pool_output_size or in_channels+out_channels)
        outputs = torch.sigmoid(self.outputs_weight * outputs) * outputs
        outputs = torch.cat((inputs, outputs), dim=2)

        if self.pool:
            outputs = self.pool(outputs)

        return outputs