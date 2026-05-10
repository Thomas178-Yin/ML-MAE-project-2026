import torch
import torch.nn as nn

# 这个里面文件估计也需要改改路径引用
from models.code_base.model.iTransformer import Model as iTransformerModel

class ConfigWrapper:
    """
    将字典转换为对象属性
    """
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

class iTransformer(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        #  将 YAML 传进来的 dict 转换为对象
        self.configs = ConfigWrapper(config_dict['model'])
        
        # 实例化iTransformer
        self.model = iTransformerModel(self.configs)
        
    def forward(self, x):
        """
        输入 x shape: [batch_size, T, N] (B, 序列长度/时间点, 通道数/特征数)
        """
        x = x.permute(0,2,1)
        B, T, N = x.shape
        device = x.device

        x_enc = x 
        
        # 构造虚拟的时间戳 (iTransformer 内置使用)
        mark_dim = 4 
        x_mark_enc = torch.zeros(B, self.configs.seq_len, mark_dim).float().to(device)
        
        # ---------------------------------------------------------
        # 占位符
        # 分类或异常检测任务不需要 x_dec 和 x_mark_dec
        # ---------------------------------------------------------
        if self.configs.task_name in ['long_term_forecast', 'short_term_forecast', 'imputation']:
            pred_len = getattr(self.configs, 'pred_len', 0)
            label_len = getattr(self.configs, 'label_len', 0)
            
            dec_inp_zeros = torch.zeros([B, pred_len, N]).float().to(device)
            # 取序列最后的 label_len 步
            dec_inp_label = x[:, -label_len:, :] if label_len > 0 else torch.empty((B, 0, N), device=device)
            x_dec = torch.cat([dec_inp_label, dec_inp_zeros], dim=1) # [B, label_len + pred_len, N]
            
            x_mark_dec = torch.zeros(B, label_len + pred_len, mark_dim).float().to(device)
        else:
            # Classification 
            x_dec = None
            x_mark_dec = None
        
        # 调用 iTransformer
        output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
        return output