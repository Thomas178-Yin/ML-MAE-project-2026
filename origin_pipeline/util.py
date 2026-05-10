import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

#——————————————————————————————————————————————————————
# 随机种子固定
#——————————————————————————————————————————————————————
def set_seed(seed: int = 42):
    """
    固定所有可控的随机种子，保证实验可复现
    """
    # Python 自带随机
    random.seed(seed)

    # numpy 随机
    np.random.seed(seed)

    # PyTorch CPU 随机
    torch.manual_seed(seed)

    # PyTorch GPU 随机（当前 GPU）
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU

    # 保证 CUDA 算子确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#——————————————————————————————————————————————————————
# 计算分类指标
#——————————————————————————————————————————————————————
def compute_metrics(y_true, y_pred_logits):
    """
    分类任务指标计算
    :param y_true: Tensor (B,)
    :param y_pred_logits: Tensor (B, num_classes)
    """
    y_true = y_true.detach().cpu().numpy()
    
    # 将模型输出的 logits 转为预测类别
    preds = torch.argmax(y_pred_logits, dim=1).detach().cpu().numpy()

    acc = accuracy_score(y_true, preds)
    # 对于不平衡数据，macro 平均比较常用
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average='macro', zero_division=0
    )

    return {
        "ACC": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

def init_model(model_class, config):
    if model_class.__name__ == "EEGNet":
        return model_class(
            chans=config['model']['chans'],
            num_classes=config['model']['num_classes'],
            time_point=config['model']['time_point']
        ).to(config['train']['device'])
    elif model_class.__name__ in ["EEGGRU", "iTransformer", "PatchTST", "TimesNet"]:
        return model_class(config).to(config['train']['device'])
    else:
        raise ValueError(f"[Error] 未知的模型: {model_class.__name__}，请在 utils.py 的 init_model 中添加对应的初始化逻辑！")

















