import torch
import torch.nn as nn

from models.code_base.model.PatchTST import Model as PatchTSTModel

class ConfigWrapper:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

class PatchTST(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        self.configs = ConfigWrapper(config_dict['model'])
        self.model = PatchTSTModel(self.configs)

        for p in self.model.parameters():
            p.requires_grad = True
            
    def forward(self, x):
        """
        输入 x.T shape: [B, T, N]
        """
        x = x.permute(0,2,1)
        return self.model(x, x_mark_enc=None, x_dec=None, x_mark_dec=None)