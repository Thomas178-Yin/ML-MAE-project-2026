import os
import torch
import torch.nn as nn

# 请根据实际路径调整导入
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
        # 将 YAML 传进来的 dict 转换为对象
        self.config_dict = config_dict
        self.configs = ConfigWrapper(config_dict['model'])
        device = config_dict["train"]["device"]
        
        # 实例化骨干网络
        self.model = CBraModModel(  
            in_dim = self.configs.in_dim, 
            out_dim = self.configs.out_dim, 
            d_model = self.configs.d_model, 
            dim_feedforward = self.configs.dim_feedforward, 
            n_layer = self.configs.n_layer,
            nhead   = self.configs.nhead
        )
        
        c = self.configs.ch_num
        T_all = self.configs.T_all  
        self.class_num = self.configs.num_classes

        #  [B, C, T_all] - [B, C * T_all] 
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * T_all, self.class_num),
        )
        
        # 冻结与权重加载逻辑
        self.weight_path = self.configs.weight_path

        if self.weight_path and os.path.exists(self.weight_path):
            ckpt = torch.load(self.weight_path, map_location=device, weights_only=True)
            self.model.load_state_dict(ckpt, strict=False)
            print("✅ Pretrained Backbone Weights Loaded.")

        # 冻结骨干网络
        for p in self.model.parameters():
            p.requires_grad = False

        for p in self.fc.parameters():
            p.requires_grad = True


    def forward(self, x):
        """
        统一的动态拆分前向传播
        输入 x: [B, C, T_all]
        """
        B, C, T_all = x.shape
        
        assert T_all % 200 == 0, f"[Error] CBraMod 要求时间长度必须是 200 的整数倍，当前数据长度为 {T_all}"
        
        # patch_num
        patch_num = T_all // 200
        
        # [B, C, T_all] -> [B, C, patch_num, 200]
        x = x.reshape(B, C, patch_num, 200)

        # [B, C, patch_num, 200]
        z = self.model(x)

        # [B, C, patch_num, 200] -> [B, C, T_all]
        z = z.reshape(B, C, -1)

        logits = self.fc(z)
        
        return logits

    def return_training_parameters(self):
        training_parameters = []
        # training_parameters.append(
        #             {
        #         "params": self.model.parameters(),
        #         "lr": self.config_dict['train']['lr'] * 0.1,
        #         "weight_decay": self.config_dict['train'].get('weight_decay', 1.0e-4)
        #     }
        # )
        training_parameters.append(
                    {
                "params": self.fc.parameters(),
                "lr": self.config_dict['train']['lr'],
                "weight_decay": self.config_dict['train'].get('weight_decay', 1.0e-4)
            }
        )
        return training_parameters