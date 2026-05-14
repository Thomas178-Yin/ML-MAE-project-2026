import os
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

# 替换为你实际的相对路径
from models.code_base.model.EEGPT import EEGTransformer
from models.code_base.model.EEGPT2 import Conv1dWithConstraint

class ConfigWrapper:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

class EEGPT(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        self.config_dict = config_dict
        self.configs = ConfigWrapper(config_dict['model'])
        self.device = config_dict["train"]["device"]
        
        in_channels = self.configs.in_channels
        use_channels_names = self.configs.use_channels_names
        target_chans = len(use_channels_names)
        self.class_num = self.configs.class_num
        
        # --- 时间对齐核心参数 ---
        self.desired_time_len = self.configs.desired_time_len 
        self.patch_size = self.configs.patch_size            
        
        # 1. 空间通道映射 (例如 22 -> 19)
        if getattr(self.configs, 'use_chan_conv', True):
            self.chan_conv = Conv1dWithConstraint(in_channels, target_chans, kernel_size=1, max_norm=1.0)
        else:
            self.chan_conv = nn.Identity()

        # 2. 大模型骨干网络
        self.target_encoder = EEGTransformer(
            img_size=self.configs.img_size,       
            patch_size=self.patch_size, 
            embed_num=self.configs.embed_num,                            
            embed_dim=self.configs.embed_dim,                    
            depth=self.configs.depth,            
            num_heads=self.configs.num_heads,
            mlp_ratio=4.0,
            drop_rate=0.1,       
            attn_drop_rate=0.1,
            drop_path_rate=self.configs.enc_drop_path_rate, 
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        self.chans_id = self.target_encoder.prepare_chan_ids(use_channels_names)
        
        # 3. 下游分类器
        # 骨干输出的维度为 [B, N, embed_num, embed_dim]
        # 计算将要展平的总特征长度
        N = self.desired_time_len // self.patch_size
        flatten_dim = N * self.configs.embed_num * self.configs.embed_dim
        d_model = self.configs.embed_dim
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, d_model * 2),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model * 2, self.class_num)
        )

        # 4. 加载预训练权重
        weight_path = getattr(self.configs, 'weight_path', None)
        if weight_path and os.path.exists(weight_path):
            ckpt = torch.load(weight_path, map_location='cpu', weights_only=False) 
            state_dict = ckpt.get('state_dict', ckpt)
            enc_weights = {k[15:]: v for k, v in state_dict.items() if k.startswith("target_encoder.")}
            self.target_encoder.load_state_dict(enc_weights, strict=False)
            print("✅ EEGPT Pretrained Weights Loaded.")

        # 5. 冻结策略 
        if getattr(self.configs, 'use_freeze_encoder', False):
            for p in self.target_encoder.parameters():
                p.requires_grad = True
                print(p.requires_grad)

    def _temporal_interpolation(self, x):
        """执行插值"""
        # x shape: [B, C, T]
        if getattr(self.configs, 'use_mean_pooling', False):
            x = x - torch.mean(x, dim=-1, keepdim=True)
        # 线性插值
        x = F.interpolate(x, size=self.desired_time_len, mode='linear', align_corners=False)
        return x

    def forward_features(self, x):
        if x.shape[-1] != self.desired_time_len:
            x = self._temporal_interpolation(x)
            
        x = self.chan_conv(x)
        z = self.target_encoder(x, self.chans_id.to(x.device)) 
        return z

    def forward(self, x):
        # 提取特征 -> [B, N, embed_num, embed_dim]
        feats = self.forward_features(x)
        
        # 分类器 (内部自动 Flatten)
        logits = self.classifier(feats) 
        return logits
    
    def return_training_parameters(self):
        training_parameters = []
        training_parameters.append(
                    {
                "params": self.target_encoder.parameters(),
                "lr": self.config_dict['train']['lr'] * 0.05,
                "weight_decay": self.config_dict['train'].get('weight_decay', 1.0e-4)
            }
        )
        training_parameters.append(
                    {
                "params": self.classifier.parameters(),
                "lr": self.config_dict['train']['lr'],
                "weight_decay": self.config_dict['train'].get('weight_decay', 1.0e-4)
            }
        )
        return training_parameters