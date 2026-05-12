import os
import sys
import torch
import yaml
import argparse
import pandas as pd

from util import set_seed, path_solution, config_fix, val_matrix
#——————————————————
# 解决导入路径
#——————————————————
path_solution(__file__)

from dataset import get_dataloader
from models.model_EEGNet import EEGNet
from models.model_iTransformer import iTransformer
from models.model_PatchTST import PatchTST
from models.model_TimesNet import TimesNet
from models.model_EEGGRU import EEGGRU
from models.model_CBraMod import CBraMod
from models.model_LaBraM import LaBraM
from models.model_EEGPT import EEGPT
from train import train, test

#——————————————————
# 选择数据集
#——————————————————
DATASET_LIST = ['MDD', 'BCIC2A', 'CHINESE', 'SEED', 'SLEEP']
dataset_id = 0
dataset_name = DATASET_LIST[dataset_id]

MODEL_LIST = [EEGNet, EEGGRU, iTransformer, PatchTST, TimesNet, CBraMod, LaBraM, EEGPT]
#——————————————————
# 选择模型
#——————————————————
model_id = 5
model = MODEL_LIST[model_id]
model_name = MODEL_LIST[model_id].__name__
print(model_name)
#——————————————————
# 设置随机参数
#——————————————————
seed = 42

DATASET_PATH = f"C:/Users/ASUS/Desktop/machine_learning/course_project/{dataset_name}"  

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default=DATASET_PATH , help='data_dir')

parser.add_argument('--model_save_dir', type=str, default=f"C:/Users/ASUS/Desktop/machine_learning/course_project/results/{dataset_name}/{model_name}", help='model save dir')
args = parser.parse_args()
print(args)

# 检查并创建保存目录
os.makedirs(args.model_save_dir, exist_ok=True)

project_root = os.path.dirname(os.path.dirname(__file__))  # 适用于模块化项目
config_path = os.path.join(project_root, "origin_pipeline", "config", f"base_{model_name}.yaml")

with open(config_path, "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)
    # 提取需要的键
    keys = ["train", "model"]
    # 使用列表推导式 + join() 安全拼接, 
    output = "\n".join(f"{key}: {config.get(key, '')}" for key in keys)
    print(output)

# 针对数据集修复config参数
config = config_fix(config, dataset_name, model_name)

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

best_model, best_val_acc = train(
                            model            = model,
                            train_loader     = dataloader_train,
                            val_loader       = dataloader_val, 
                            dataset          = dataset_name,    
                            config           = config,
                            seed             = seed,
                                )

predictions = test(
                    best_model_state= best_model,
                    model           = model,    
                    config          = config, 
                    test_loader     = dataloader_test,
                    # seed            = seed,
                    )

# 保存模型
save_path = os.path.join(args.model_save_dir, f"best_model.pt")
torch.save(best_model, save_path)
print(f"[✔] Model saved to {save_path}")

# ————————————————————
# 保存预测结果为 CSV
# ————————————————————
csv_save_path = os.path.join(args.model_save_dir, f"predictions_{dataset_name}_{model_name}.csv")
df_preds = pd.DataFrame({"Prediction": predictions})
df_preds.to_csv(csv_save_path, index=False)
print(f"[✔] Predictions saved to {csv_save_path}")

# ————————————————————
# 汇总val结果
# ————————————————————
val_matrix(
    val_acc = best_val_acc,
    model_name = model_name,
    dataset_name = dataset_name,
    model_save_dir = args.model_save_dir
)