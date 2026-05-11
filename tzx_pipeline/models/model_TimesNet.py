import torch
import torch.nn as nn

from models.code_base.model.TimesNet import Model as TimesNetModel

class ConfigWrapper:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

class TimesNet(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        self.configs = ConfigWrapper(config_dict['model'])
        self.model = TimesNetModel(self.configs)

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = x.permute(0,2,1)
        B, T, N = x.shape
        device = x.device
        
        x_mark_mask = torch.ones(B, T).float().to(device)

        # 调用 TimesNet
        return self.model(x, x_mark_mask, x_dec=None, x_mark_dec=None)