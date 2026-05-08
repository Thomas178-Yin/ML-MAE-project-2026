import os
import sys
import torch
import yaml
import argparse

from utils import set_seed
from dataset import get_dataloader
from model.model_EEGNet import EEGNet
from train import train, test

#——————————————————
# 解决导入路径
#——————————————————
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
add_path = os.path.join(BASE_DIR, "origin_pipeline")
sys.path.append(add_path)

DATASET_LIST = ['MDD', 'BCIC2A', 'CHINESE', 'SEED', 'SLEEP']
#——————————————————
# 选择数据集
#——————————————————
dataset_id = 0
dataset_name = DATASET_LIST[dataset_id]
print
MODEL_LIST = [EEGNet]
#——————————————————
# 选择模型
#——————————————————
model_id = 0
model = MODEL_LIST[model_id]
model_name = MODEL_LIST[model_id].__name__
print(model_name)
#——————————————————
# 设置随机参数
#——————————————————
seed = 42

DATASET_PATH = f"E:/G2/machine_learning/code/12_RNN/course_project/{dataset_name}"

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default=DATASET_PATH , help='data_dir')

parser.add_argument('--model_save_dir', type=str, default="None/%s" % dataset_name, help='model save dir')
args = parser.parse_args()
print(args)

# 检查并创建保存目录
os.makedirs(args.model_save_dir, exist_ok=True)

project_root = os.path.dirname(os.path.dirname(__file__))  # 适用于模块化项目
config_path = os.path.join(project_root, "origin_pipline", "config", f"base_{model_name}.yaml")

with open(config_path, "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)
    # 提取需要的键
    keys = ["train", "model"]
    # 使用列表推导式 + join() 安全拼接, 
    output = "\n".join(f"{key}: {config.get(key, '')}" for key in keys)
    print(output)

# 保存yaml文件到模型保存路径 方便查看
save_config_path = os.path.join(args.model_save_dir, f"config_used_{dataset_name}_{model_name}.yaml")
with open(save_config_path, "w") as f:
    yaml.dump(config, f)
print(f"config save to: {save_config_path}")


#————————————————————
#训练过程
#————————————————————

# 固定种子
set_seed(seed = seed)

dataloader_train, dataloader_val, dataloader_test = get_dataloader(args = args, batch_size = config['dataset']['batch_size'], seed = seed)

print("Num batches:")
print(f"  Train: {len(dataloader_train)}")
print(f"  Val:   {len(dataloader_val)}")
print(f"  Test:  {len(dataloader_test)}")

best_model = train(
                   model            = model,
                   train_loader     = dataloader_train,
                   val_loader       = dataloader_val,     
                   config           = config,
                   seed             = seed,
                    )

total_metrics = test(
                    best_model      = best_model,
                    model           = model,    
                    config          = config, 
                    seed            = seed,

                    )

# 保存模型
save_path = os.path.join(args.model_save_dir, f"None.pt")
torch.save(best_model, save_path)
print(f"[✔] Model saved to {save_path}")