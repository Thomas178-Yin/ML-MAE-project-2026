import torch
import torch.nn as nn

class ConfigWrapper:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

class EEGGRU(nn.Module):
    def __init__(self, config_dict):
        super(EEGGRU, self).__init__()
        self.config_dict = config_dict
        # 1. 解析配置
        conf = ConfigWrapper(config_dict['model'])
        self.hidden_size = conf.hidden_size
        self.num_layers = conf.num_layers
        
        # 2. 定义 GRU 层
        # input_channels 对应 EEG 通道数 (N)，time_points 对应时间步长 (T)
        self.gru = nn.GRU(
            input_size=conf.input_channels,
            hidden_size=conf.hidden_size,
            num_layers=conf.num_layers,
            batch_first=True,
            dropout=conf.dropout if conf.num_layers > 1 else 0.0
        )
        
        # 3. 全连接分类层
        self.fc = nn.Sequential(
            nn.Dropout(conf.dropout),
            nn.Linear(conf.hidden_size, conf.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(conf.hidden_size // 2, conf.num_classes)
        )

        for p in self.gru.parameters():
            p.requires_grad = True
        
        for p in self.fc.parameters():
            p.requires_grad = True
        
        
    def forward(self, x):
        # 输入 x 形状: (Batch, Channels, Time)
        # 转换为 GRU 期望的形状: (Batch, Time, Channels)
        x = x.permute(0, 2, 1)  
        
        # out: (Batch, Time, Hidden_size)
        out, _ = self.gru(x)
        
        # 取最后一个时间步的输出
        last_out = out[:, -1, :]  
        
        # 分类输出
        logits = self.fc(last_out)
        return logits
    
    def return_training_parameters(self):
        training_parameters = []
        training_parameters.append(
                    {
                "params": self.gru.parameters(),
                "lr": self.config_dict['train']['lr'],
                "weight_decay": self.config_dict['train'].get('weight_decay', 1.0e-4)
            }
        )
        training_parameters.append(
                    {
                "params": self.fc.parameters(),
                "lr": self.config_dict['train']['lr'],
                "weight_decay": self.config_dict['train'].get('weight_decay', 1.0e-4)
            }
        )
        return training_parameters