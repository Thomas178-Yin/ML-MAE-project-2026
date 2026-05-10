import os

import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

from models.code_base.model.CBraMod import CBraMod as CBraModModel

class ConfigWrapper:
    """
    将字典转换为对象属性
    """
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

class CBraMod(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        #  将 YAML 传进来的 dict 转换为对象
        self.configs = ConfigWrapper(config_dict['model'])
        
        # 实例化iTransformer
        self.model = CBraModModel(  in_dim  = self.configs.in_dim, 
                                    out_dim = self.configs.out_dim, 
                                    d_model = self.configs.d_model, 
                                    dim_feedforward = self.configs.dim_feedforward, 
                                    n_layer = self.configs.n_layer,
                                    nhead   = self.configs.nhead
                                    ).to(config_dict["train"]["device"])

        c = self.configs.ch_num
        s = self.configs.patch_num
        p = self.configs.patch_size
        self.class_num = self.configs.num_classes

        self.fc = nn.Sequential(
                                Rearrange('b c p -> b (c p)'),
                                nn.Linear(c*p , self.class_num),
                            ).to(config_dict["train"]["device"])
        
        # 加载权重
        self.weight_path = config_dict['model']['weight_path']

        if self.weight_path and os.path.exists(self.weight_path):
            ckpt = torch.load(self.weight_path, map_location = config_dict["train"]["device"], weights_only=True)
            self.model.load_state_dict(ckpt, strict=False)
            print("✅ Pretrained Backbone Weights Loaded.")

        for p in self.model.parameters():
            p.requires_grad = False

        for p in self.fc.parameters():
            p.requires_grad = True


    def forward(self, x):
        B, C, T_all = x.shape
        x = x.reshape(4, -1, C, T_all).permute(1,2,0,3)
        if x.dim() == 3:
            x = x.unsqueeze(2) # [1, 62, 800] -> [1, 62, 1, 800]

        z = self.model(x)

        z = z.permute(2,0,1,3).reshape(B, C, -1)

        logits = self.fc(z)
        return logits