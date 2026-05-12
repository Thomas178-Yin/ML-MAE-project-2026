import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from util import compute_metrics, init_model, set_seed


def train(  model,
            train_loader,
            val_loader,
            dataset,
            config,
            seed = 42):
    
    run_dir = f"origin_pipeline/log/{dataset}/{model.__name__}/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    writer = SummaryWriter(log_dir=run_dir)

    device = config['train']['device']
    set_seed(seed)
    
    model = init_model(model, config)

    # for p in model.parameters():
    #         p.requires_grad = True

    optimizer = optim.Adam(
                        model.return_training_parameters(),
                        weight_decay=config['train'].get('weight_decay', 1.0e-4)
                        )

    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # 初始化
    # -------------------------
    train_losses = []
    val_losses = []
    val_accuracies = []

    patience = config['train']["early_stop_patience"]
    min_delta = config['train']["early_stop_min_delta"]

    best_val_acc = None
    best_model = None
    bad_epochs = 0
    
    train_loss_sum = 0.0
    train_num = 0

    for epoch in tqdm(range(config['train'].get("epochs", 100)), desc="Epochs", position=0, leave=False):
        model.train()

        epoch_loss = 0.0

        for data, label in tqdm(train_loader, desc="Batches", position=1, leave=False):
            data    = data.to(device)
            label   = label.to(device)

            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            batch_size = label.size(0)
            train_loss_sum += loss.item() * batch_size
            train_num += batch_size
            epoch_loss += loss.item()


        epoch_train_loss = train_loss_sum / train_num
        train_losses.append(epoch_train_loss)

        num_batches = len(train_loader)
        avg_epoch_loss = epoch_loss / num_batches

        writer.add_scalar("Train/loss", avg_epoch_loss, epoch)

        # ================= 验证阶段 =================
        with torch.no_grad():
            total_metrics = evaluate(
                                    model       = model,
                                    val_loader  = val_loader,
                                    criterion   = criterion,
                                    device      = device
                                 )
        
        val_loss    = total_metrics["Loss"]
        val_acc     = total_metrics["ACC"]

        writer.add_scalar("Val/loss",   val_loss,   epoch)
        writer.add_scalar("Val/ACC",    val_acc,    epoch)

        if best_val_acc is None or (val_acc - best_val_acc) > min_delta:
            best_val_acc = val_acc
            bad_epochs = 0

            best_model = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
            }
            print(f"[✔] New best model at epoch {epoch+1}, Val_ACC={best_val_acc:.4f}")
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    writer.close()
    return best_model, best_val_acc

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            batch_size = label.size(0)
            total_loss += loss.item() * batch_size

            all_preds.append(output)
            all_labels.append(label)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 计算指标
    metrics = compute_metrics(all_labels, all_preds)
    # 计算整个 epoch 的平均 loss
    metrics["Loss"] = total_loss / len(val_loader.dataset)

    return metrics

def test(best_model_state, model, test_loader, config):
    device = config['train']['device']
    
    # 实例化模型
    model = init_model(model, config)
    
    # 挂载最优权重
    model.load_state_dict(best_model_state['model'])
    model.to(device)
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            
            output = model(data)
            pred_class = torch.argmax(output, dim=1)
            predictions.extend(pred_class.cpu().numpy())
            
    return predictions