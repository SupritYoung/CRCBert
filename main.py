#!/usr/bin/env python
# coding=utf-8

import torch
import random
import argparse
import sys
import warnings
import logging
import os
import json
from tqdm import tqdm, trange
from time import strftime, localtime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F

from data_utils import load_dataset
from models import *

warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='/root/models/mc-bert', help='模型名称')
    parser.add_argument('--device', type=str, default="cuda", help='gpu or cpu')
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. eg: 16, 32, 64")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate. eg: 2e-5, 1e-5")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs.")
    parser.add_argument("--seed", type=int, default=1000,
                        help="Random seed.")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                        help='option: train, test')
    parser.add_argument('--data_dir', default='datas/', type=str, required=False, help='数据集路径')
    parser.add_argument('--model_dir', default='checkpoints/', type=str, required=False, help='模型保存路径')

    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# def masked_bce_loss(output, target, mask):
#     """
#     计算带有掩码的 BCE Loss，只对非缺失值进行计算
#     :param output: 模型输出
#     :param target: 目标值
#     :param mask: 掩码矩阵，0 表示缺失值，1 表示有效值
#     :return: 平均 BCE Loss
#     """
#     loss = F.binary_cross_entropy_with_logits(output, target, reduction='none')
#     loss = loss * mask
#     loss = loss.sum() / mask.sum()
#     return loss

def train(args, model, train_loader, val_loader, test_loader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(reduction='none')  # 使用基础的 BCEWithLogitsLoss

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in trange(args.epochs, desc="Epoch"):
        model.train()
        train_loss = 0.0
        all_train_labels = []
        all_train_preds = []
        train_errors = []

        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            outputs = model(
                record_input_ids=batch['record_input_ids'].to(args.device),
                record_attention_mask=batch['record_attention_mask'].to(args.device),
                mri_input_ids=batch['mri_input_ids'].to(args.device),
                mri_attention_mask=batch['mri_attention_mask'].to(args.device),
                ct_input_ids=batch['ct_input_ids'].to(args.device),
                ct_attention_mask=batch['ct_attention_mask'].to(args.device)
            )
            labels = batch['labels'].to(args.device)
            mask = ~torch.isnan(labels)  # 创建掩码矩阵
            labels = torch.where(mask, labels, outputs.detach())  # 将 nan 值替换为预测输出值
            loss = criterion(outputs, labels)
            loss = (loss * mask).sum() / mask.sum()  # 计算掩码后的平均损失
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            labels = labels.cpu().numpy()
            mask_np = mask.cpu().numpy().astype(bool)
            
            all_train_labels.append(labels[mask_np])
            all_train_preds.append(preds[mask_np])
            
            # 收集错误预测
            for i in range(len(labels)):
                if not np.array_equal(labels[i][mask_np[i]], (preds[i][mask_np[i]] > 0.5).astype(int)):
                    train_errors.append({
                        'patient_SN': batch['patient_SN'][i],
                        # 'record': batch['record'][i],
                        # 'MRI': batch['MRI'][i],
                        # 'CT': batch['CT'][i],
                        'Ki-67': labels[i][0], 'Ki-67-pred': (preds[i][0] > 0.5).astype(int),
                        'MSI': labels[i][1], 'MSI-pred': (preds[i][1] > 0.5).astype(int),
                        'CK': labels[i][2], 'CK-pred': (preds[i][2] > 0.5).astype(int),
                        'P53': labels[i][3], 'P53-pred': (preds[i][3] > 0.5).astype(int),
                    })

        avg_train_loss = train_loss / len(train_loader)

        all_train_labels = np.concatenate(all_train_labels, axis=0)
        all_train_preds = np.concatenate(all_train_preds, axis=0)
        all_train_preds_binary = (all_train_preds > 0.5).astype(int)

        train_accuracy = accuracy_score(all_train_labels, all_train_preds_binary)
        train_precision = precision_score(all_train_labels, all_train_preds_binary, average='macro', zero_division=0)
        train_recall = recall_score(all_train_labels, all_train_preds_binary, average='macro', zero_division=0)
        train_f1 = f1_score(all_train_labels, all_train_preds_binary, average='macro', zero_division=0)

        logging.info(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}')
        
        val_loss, val_metrics, val_errors = eval(args, model, val_loader)
        logging.info(f'Epoch [{epoch+1}/{args.epochs}], Validation Loss: {val_loss:.4f}')
        logging.info(f'Validation Metrics: {val_metrics}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        # 保存错误预测
        train_errors_df = pd.DataFrame(train_errors)
        train_errors_df.to_csv(os.path.join(args.model_dir, 'train_errors.csv'), index=False)
        val_errors_df = pd.DataFrame(val_errors)
        val_errors_df.to_csv(os.path.join(args.model_dir, 'val_errors.csv'), index=False)
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pth'))
    
    test_loss, test_metrics, _ = eval(args, model, test_loader)
    logging.info(f'Test Loss: {test_loss:.4f}')
    logging.info(f'Test Metrics: {test_metrics}')

def eval(args, model, data_loader):
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction='none')  # 使用基础的 BCEWithLogitsLoss
    eval_loss = 0.0

    all_labels = []
    all_preds = []
    all_errors = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            outputs = model(
                record_input_ids=batch['record_input_ids'].to(args.device),
                record_attention_mask=batch['record_attention_mask'].to(args.device),
                mri_input_ids=batch['mri_input_ids'].to(args.device),
                mri_attention_mask=batch['mri_attention_mask'].to(args.device),
                ct_input_ids=batch['ct_input_ids'].to(args.device),
                ct_attention_mask=batch['ct_attention_mask'].to(args.device)
            )
            labels = batch['labels'].to(args.device)
            mask = ~torch.isnan(labels)  # 创建掩码矩阵
            labels = torch.where(mask, labels, outputs)  # 将 nan 值替换为预测输出值
            loss = criterion(outputs, labels)
            loss = (loss * mask).sum() / mask.sum()  # 计算掩码后的平均损失
            eval_loss += loss.item()

            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            labels = labels.cpu().numpy()
            mask_np = mask.cpu().numpy().astype(bool)

            # 只保留掩码为1的标签和预测值
            all_labels.append(labels[mask_np])
            all_preds.append(preds[mask_np])

            # 保存错误预测
            for i in range(len(labels)):
                if not np.array_equal(labels[i][mask_np[i]], (preds[i][mask_np[i]] > 0.5).astype(int)):
                    all_errors.append({
                        'patient_SN': batch['patient_SN'][i],
                        # 'record': batch['record'][i],
                        # 'MRI': batch['MRI'][i],
                        # 'CT': batch['CT'][i],
                        'Ki-67': labels[i][0], 'Ki-67-pred': (preds[i][0] > 0.5).astype(int),
                        'MSI': labels[i][1], 'MSI-pred': (preds[i][1] > 0.5).astype(int),
                        'CK': labels[i][2], 'CK-pred': (preds[i][2] > 0.5).astype(int),
                        'P53': labels[i][3], 'P53-pred': (preds[i][3] > 0.5).astype(int),
                    })

    avg_eval_loss = eval_loss / len(data_loader)
    
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # 将预测结果二值化
    all_preds_binary = (all_preds > 0.5).astype(int)

    accuracy = accuracy_score(all_labels, all_preds_binary)
    precision = precision_score(all_labels, all_preds_binary, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds_binary, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds_binary, average='macro', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    return avg_eval_loss, metrics, all_errors



if __name__ == '__main__':
    args = init_args()
    logging.info(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_seed(args.seed)

    # 加载数据
    train_loader, val_loader, test_loader = load_dataset(args)
    # print(next(iter(train_loader)))  # 打印一个训练批次的样本

    # 初始化模型
    model = initialize_model(args.model_name, args.device)

    # 训练
    train(args, model, train_loader, val_loader, test_loader)
    # 评估
    eval(args, model, val_loader)