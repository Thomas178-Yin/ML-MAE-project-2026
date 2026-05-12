import os
import torch
import torch.nn as nn
from collections import OrderedDict
from timm import create_model


from models.code_base.model.LaBraM import NeuralTransformer

class ConfigWrapper:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

class LaBraM(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        self.config_dict = config_dict
        self.configs = ConfigWrapper(config_dict['model'])
        self.device = config_dict["train"]["device"]

        self.chans_num = self.configs.ch_num
        self.class_num = self.configs.num_classes
        
        # 使用 config_fix 算好的总时间点
        EEG_size = self.configs.EEG_size  
        
        # 实例化大模型
        self.target_encoder = create_model(
            "labram_base_patch200_200", 
            EEG_size=EEG_size,                  
            qkv_bias=False,
            rel_pos_bias=True,
            in_chans=1,                         
            out_chans=8,                        
            num_classes= 0,  
            drop_rate=0.0,
            drop_path_rate=0.1,
            use_rel_pos_bias=False,
            use_abs_pos_emb=True,
            init_values=0.1,  
        )
        
        # 加载预训练权重 
        weight_path = getattr(self.configs, 'weight_path', None)
        if weight_path and os.path.exists(weight_path):
            print(f"Loading LaBraM pretrained weights from {weight_path}")
            checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
            checkpoint_model = checkpoint.get('model', checkpoint.get('module', checkpoint))
            
            # 处理 'student.' 前缀 
            new_dict = OrderedDict()
            for k, v in checkpoint_model.items():
                if k.startswith('student.'): k = k[8:]
                new_dict[k] = v
            
            # 剔除形状不匹配的层
            state_dict = self.target_encoder.state_dict()
            for k in list(new_dict.keys()):
                if k in state_dict and new_dict[k].shape != state_dict[k].shape:
                    new_dict.pop(k)
                if "relative_position_index" in k:
                    new_dict.pop(k)

            self.target_encoder.load_state_dict(new_dict, strict=False)
            print("✅ LaBraM Pretrained Weights Loaded.")

        # 分类头
        d_model = self.configs.d_model
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, self.class_num),
        )
        self.target_encoder.head = nn.Identity()

        # 冻结骨干 
        for name, p in self.target_encoder.named_parameters():
            # 参数名字里带有 'time_embed' 或 'pos_embed'，就让它计算梯度
            if "time_embed" in name:# or "pos_embed" in name:
                p.requires_grad = True
                print(f"🔓 Unfreezing {name} to learn new sequence length.")
            else:
                p.requires_grad = False

    def forward_features(self, x):
        B, C, T_all = x.shape

        assert T_all % 200 == 0, f"LaBraM requires T to be a multiple of 200, got {T_all}"
        patch_num = T_all // 200
        x = x.reshape(B, C, patch_num, 200) 
        
        z = self.target_encoder.forward_features(x, input_chans=[i for i in range(C+1)])
        return z

    def forward(self, x):
        feats = self.forward_features(x)  
        logits = self.classifier(feats) 
        return logits
    
    def return_training_parameters(self):
        training_parameters = []
        # training_parameters.append(
        #             {
        #         "params": self.target_encoder.parameters(),
        #         "lr": self.config_dict['train']['lr'] * 0.1,
        #         "weight_decay": self.config_dict['train'].get('weight_decay', 1.0e-4)
        #     }
        # )

        training_parameters.append(
                    {
                "params": self.classifier.parameters(),
                "lr": self.config_dict['train']['lr'],
                "weight_decay": self.config_dict['train'].get('weight_decay', 1.0e-4)
            }
        )
        
        return training_parameters