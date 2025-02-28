import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

class CancerDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        record_text = row['record']
        mri_text = row['MRI'] if pd.notnull(row['MRI']) else ''
        ct_text = row['CT'] if pd.notnull(row['CT']) else ''
        labels = row[['Ki-67', 'MSI', 'CK', 'P53']].values.astype(float)

        record_encoding = self.tokenizer.encode_plus(
            record_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        mri_encoding = self.tokenizer.encode_plus(
            mri_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        ct_encoding = self.tokenizer.encode_plus(
            ct_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'patient_SN': row['patient_SN'],
            'record_input_ids': record_encoding['input_ids'].squeeze(0),
            'record_attention_mask': record_encoding['attention_mask'].squeeze(0),
            'mri_input_ids': mri_encoding['input_ids'].squeeze(0),
            'mri_attention_mask': mri_encoding['attention_mask'].squeeze(0),
            'ct_input_ids': ct_encoding['input_ids'].squeeze(0),
            'ct_attention_mask': ct_encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

def collate_fn(batch):
    patient_SN = [item['patient_SN'] for item in batch]
    record_input_ids = torch.stack([item['record_input_ids'] for item in batch])
    record_attention_mask = torch.stack([item['record_attention_mask'] for item in batch])
    mri_input_ids = torch.stack([item['mri_input_ids'] for item in batch])
    mri_attention_mask = torch.stack([item['mri_attention_mask'] for item in batch])
    ct_input_ids = torch.stack([item['ct_input_ids'] for item in batch])
    ct_attention_mask = torch.stack([item['ct_attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'patient_SN': patient_SN,
        'record_input_ids': record_input_ids,
        'record_attention_mask': record_attention_mask,
        'mri_input_ids': mri_input_ids,
        'mri_attention_mask': mri_attention_mask,
        'ct_input_ids': ct_input_ids,
        'ct_attention_mask': ct_attention_mask,
        'labels': labels
    }

def load_dataset(args):
    train_path = os.path.join(args.data_dir, 'train_datas.csv')
    val_path = os.path.join(args.data_dir, 'val_datas.csv')
    test_path = os.path.join(args.data_dir, 'test_datas.csv')

    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)

    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    train_dataset = CancerDataset(train_data, tokenizer, max_len=512)
    val_dataset = CancerDataset(val_data, tokenizer, max_len=512)
    test_dataset = CancerDataset(test_data, tokenizer, max_len=512)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
