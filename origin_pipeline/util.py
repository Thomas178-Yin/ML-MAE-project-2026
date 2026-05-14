import json
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
            config_dict = config,
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
def val_matrix(val_acc, model_name, dataset_name, model_save_dir, config):
    """
    更新实验结果汇总矩阵，并自动保存历史最优配置参数
    """
    # 路径计算：回退两级到 results 根目录 (results/Dataset/Model -> results)
    results_root = os.path.dirname(os.path.dirname(model_save_dir))
    matrix_path = os.path.join(results_root, "total_val_acc_matrix.csv")
    # 最优配置文件路径
    best_config_path = os.path.join(results_root, "best_configs.json")
    
    # 1. 加载或初始化准确率矩阵 (CSV)
    if os.path.exists(matrix_path):
        df = pd.read_csv(matrix_path, index_col=0)
    else:
        df = pd.DataFrame()

    # 2. 加载或初始化最优参数字典 (JSON)
    if os.path.exists(best_config_path):
        with open(best_config_path, 'r', encoding='utf-8') as f:
            best_configs = json.load(f)
    else:
        best_configs = {}

    # 3. 比较逻辑：获取该 (数据集, 模型) 的当前最高准确率
    current_best = 0.0
    if dataset_name in df.index and model_name in df.columns:
        val = df.at[dataset_name, model_name]
        if pd.notna(val):
            current_best = float(val)

    # 4. 如果当前准确率更高，则执行更新
    if val_acc > current_best:
        # --- 更新矩阵 ---
        df.at[dataset_name, model_name] = val_acc
        df = df.sort_index(axis=0).sort_index(axis=1)
        df.index.name = "Dataset/Model"
        df.to_csv(matrix_path)

        # --- 更新最优参数字典 ---
        if dataset_name not in best_configs:
            best_configs[dataset_name] = {}
        
        # 以字典形式存储该模型的最优参数
        best_configs[dataset_name][model_name] = {
            "best_acc": float(val_acc),
            "params": {
                "model": config.get("model", {}),
                "train": config.get("train", {})
            }
        }
        
        # 写入 JSON 文件，保持缩进方便阅读
        with open(best_config_path, 'w', encoding='utf-8') as f:
            json.dump(best_configs, f, indent=4, ensure_ascii=False)
        
        print(f"\n[🔥] 发现新纪录! {dataset_name}-{model_name} 已更新最优参数到 best_configs.json")
    else:
        print(f"\n[s] 当前结果 {val_acc:.4f} 未超过历史最高值 {current_best:.4f}，跳过参数备份。")

    # 5. 打印汇总矩阵预览
    print("\n" + "="*65)
    print(f"📊 全局最优结果矩阵 (汇总于: {matrix_path})")
    print("-" * 65)
    print(df)
    print("="*65 + "\n")

#——————————————————————————————————————————————————————
# 数值处理
#——————————————————————————————————————————————————————
def normalize_data(tensor_x, dataset_name, use_zscore=False):
    """
    根据不同数据集的原始数据分布，进行数据缩放或 Z-score 标准化。
    目标：将数据值限制在大致 [-1, 1] 之间，均值约为 0。
    
    【各数据集原始统计参考 (Raw Data Statistics)】
    ======================================================================
    MDD:
      Train   | Max:  475.80 | Min:  -507.35 | Mean:   0.0597 | Std:  14.63
      Val     | Max:  467.88 | Min:  -514.36 | Mean:   0.0134 | Std:  13.95
      Test    | Max:  587.95 | Min:  -583.81 | Mean:   0.0085 | Std:  12.76
      -> 建议: 极值约 ±600，除以 500.0

    BCIC2A:
      Train   | Max:  0.1221 | Min:  -0.1160 | Mean:   0.0014 | Std: 0.0130
      Val     | Max:  0.1171 | Min:  -0.1048 | Mean:   0.0013 | Std: 0.0134
      Test    | Max:  0.1059 | Min:  -0.1034 | Mean:   0.0008 | Std: 0.0121
      -> 建议: 极值约 ±0.12，乘以 10.0

    CHINESE:
      Train   | Max: 1821.59 | Min: -19851.7 | Mean:  -4.6865 | Std: 213.51
      Val     | Max: 7102.17 | Min: -11412.5 | Mean:  -7.2075 | Std: 248.64
      Test    | Max:     nan | Min:      nan | Mean:      nan | Std:    nan
      -> 建议: 存在极端的单侧负向离群点，方差较大。除以 200.0。并必须清理 NaN。

    SEED:
      Train   | Max:  205.91 | Min:  -192.42 | Mean:   0.0000 | Std:   5.53
      Val     | Max:  179.92 | Min:  -198.99 | Mean:   0.0000 | Std:   5.46
      Test    | Max:  202.14 | Min:  -193.47 | Mean:   0.0000 | Std:   5.51
      -> 建议: 极值约 ±200，除以 200.0

    SLEEP:
      Train   | Max: 5249.64 | Min: -4512.75 | Mean:  -0.0009 | Std:  17.00
      Val     | Max: 5616.77 | Min: -5637.30 | Mean:   0.0068 | Std:  21.33
      Test    | Max: 3309.71 | Min: -1726.25 | Mean:   0.0043 | Std:  14.56
      -> 建议: 极值约 ±5600，除以 1000.0 压制大值
    ======================================================================
    """
    
    # 将所有 NaN 替换为 0，防止 CHINESE 测试集污染后续计算
    tensor_x = torch.nan_to_num(tensor_x, nan=0.0)

    # 直接使用 Z-score 标准化
    if use_zscore:
        mean_val = tensor_x.mean()
        std_val = tensor_x.std()
        if std_val > 1e-8:
            return (tensor_x - mean_val) / std_val
        return tensor_x

    # 根据数据集自行硬编码调整 ----自己调整
    if dataset_name == 'MDD':
        tensor_x = tensor_x #/ 10.0
        
    elif dataset_name == 'BCIC2A':
        tensor_x = tensor_x #* 10.0
        
    elif dataset_name == 'CHINESE':
        # 考虑到 CHINESE 存在极端的 -19851，如果不加限制，正常信号会被压得太小
        # 这里除以 200，并对极端离群值进行截断(Clamp)
        tensor_x = tensor_x #/ 200.0
        # tensor_x = torch.clamp(tensor_x, min=-5.0, max=5.0) 
        
    elif dataset_name == 'SEED':
        tensor_x = tensor_x #/ 100.0
        
    elif dataset_name == 'SLEEP':
        tensor_x = tensor_x #/ 100.0

    return tensor_x





