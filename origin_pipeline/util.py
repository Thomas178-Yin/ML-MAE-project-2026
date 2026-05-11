import os
import random
import pandas as pd
import sys

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
    elif model_class.__name__ in ["EEGGRU", "iTransformer", "PatchTST", "TimesNet", 'CBraMod', "EEGPT", "LaBraM"]:
        return model_class(config).to(config['train']['device'])
    else:
        raise ValueError(f"[Error] 未知的模型: {model_class.__name__}，请在 utils.py 的 init_model 中添加对应的初始化逻辑！")

#——————————————————————————————————————————————————————
# 解决导入路径解决
#——————————————————————————————————————————————————————
def path_solution(path):
    BASE_DIR = os.path.dirname(os.path.abspath(path))
    add_path = os.path.join(BASE_DIR, "origin_pipeline")
    sys.path.append(add_path)

    # code-base
    code_base_path = os.path.join(BASE_DIR, "models", "code_base")
    sys.path.append(code_base_path)

    # layers
    layers_path = os.path.join(BASE_DIR, "models", "code_base", "layers")
    sys.path.append(layers_path)

    # models
    models_path = os.path.join(BASE_DIR, "models", "code_base", "model")
    sys.path.append(models_path)

    # model_
    model_path = os.path.join(BASE_DIR, "models")
    sys.path.append(model_path)

    # utils
    utils_path = os.path.join(BASE_DIR, "models", "code_base", "utils")
    sys.path.append(models_path)

#——————————————————————————————————————————————————————
# 针对数据集的参数改变
#——————————————————————————————————————————————————————
def config_fix(config, dataset_name, model_name):
    """
    根据选择的数据集自动对齐不同模型的 config 参数键值
    """
    
    # ==========================================
    # 数据集数据
    # ==========================================
    dataset_meta = {
        'MDD':     {'channels': 20, 'time': 200,  'classes': 2},
        'BCIC2A':  {'channels': 22, 'time': 800,  'classes': 4},
        'CHINESE': {'channels': 22, 'time': 200,  'classes': 2},
        'SEED':    {'channels': 62, 'time': 400,  'classes': 3},
        'SLEEP':   {'channels': 6,  'time': 6000, 'classes': 5}
    }

    if dataset_name not in dataset_meta:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    meta = dataset_meta[dataset_name]

    # ==========================================
    # 参数映射
    # ==========================================
    key_mapping = {
        'EEGNet':       ('chans',          'time_point',  'num_classes'),
        'EEGGRU':       ('input_channels', 'time_points', 'num_classes'),
        'CBraMod':      ('ch_num',         'in_dim',      'num_classes'),
        'iTransformer': ('enc_in',         'seq_len',     'num_class'),
        'PatchTST':     ('enc_in',         'seq_len',     'num_class'),
        'TimesNet':     ('enc_in',         'seq_len',     'num_class')
    }

    # 特殊处理 CBraMod
    if model_name == 'CBraMod':
        config['model']['ch_num'] = meta['channels']
        config['model']['num_classes'] = meta['classes']
        
        # patch_size 200
        config['model']['in_dim'] = 200
        config['model']['patch_size'] = 200
        
        # patch_num
        config['model']['patch_num'] = meta['time'] // 200
        # 总时间长度
        config['model']['T_all'] = meta['time'] 
        
        print(f"[*] CBraMod detected: Auto-calculated patch_num={config['model']['patch_num']} (Total Time={meta['time']}).")
    
    elif model_name == 'LaBraM':
        config['model']['ch_num'] = meta['channels']
        config['model']['num_classes'] = meta['classes']
        config['model']['in_dim'] = 200
        config['model']['patch_num'] = meta['time'] // 200
        config['model']['EEG_size'] = meta['time'] # 总长度
        print(f"[*] LaBraM: Auto-calculated patch_num={config['model']['patch_num']}")

    elif model_name == 'EEGPT':
        config['model']['in_channels'] = meta['channels']
        config['model']['class_num'] = meta['classes']
        
        # 计算 desired_time_len =
        input_T = meta['time']
        desired_time_len = int((input_T / 200) * 256)
        
        config['model']['desired_time_len'] = desired_time_len
        config['model']['T_all'] = input_T  # 记录原始长度备用
        
        target_chans = len(config['model']['use_channels_names'])
        # 保证骨干网络用插值后的目标长度初始化
        config['model']['img_size'] = [target_chans, desired_time_len] 
        
        print(f"[*] EEGPT: Time Interpolation [{input_T} -> {desired_time_len}] "
              f"({input_T}/200*256). Target Chans: {target_chans}")
    # 常规模型处理
    elif model_name in key_mapping:
        ch_key, time_key, cls_key = key_mapping[model_name]
        config['model'][ch_key] = meta['channels']
        config['model'][cls_key] = meta['classes']
        config['model'][time_key] = meta['time']
        
        if model_name == 'TimesNet' and 'c_out' in config['model']:
            config['model']['c_out'] = meta['classes']
            
        print(f"[-] Config auto-fixed for {dataset_name} + {model_name}: "
              f"Channels={meta['channels']}, Time={meta['time']}, Classes={meta['classes']}")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return config



#——————————————————————————————————————————————————————
# 保存结果
#——————————————————————————————————————————————————————
def val_matrix(val_acc, model_name, dataset_name, model_save_dir):
    """
    实验结果汇总矩阵
    """
    # 保存路径
    results_root = os.path.dirname(os.path.dirname(model_save_dir))
    matrix_path = os.path.join(results_root, "total_val_acc_matrix.csv")
    

    if os.path.exists(matrix_path):
        # 读取旧表格
        df = pd.read_csv(matrix_path, index_col=0)
    else:
        # 创建空表
        df = pd.DataFrame()

    # 更新
    df.at[dataset_name, model_name] = val_acc
    
    # 排序（可选）
    df = df.sort_index(axis=0).sort_index(axis=1)
    
    # 保存
    df.index.name = "Dataset/Model"
    df.to_csv(matrix_path)
    
    # 打印结果
    print("\n" + "="*50)
    print(f"📊 Global Result Matrix Updated (Saved to: {matrix_path})")
    print("-" * 50)
    print(df)
    print("="*50 + "\n")






