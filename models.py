import torch
import torch.nn as nn
from transformers import BertModel

# class MultiInputBERTClassifier(nn.Module):
#     def __init__(self, model_name, num_labels):
#         super(MultiInputBERTClassifier, self).__init__()
#         self.record_bert = BertModel.from_pretrained(model_name)
#         self.mri_bert = BertModel.from_pretrained(model_name)
#         self.ct_bert = BertModel.from_pretrained(model_name)
        
#         self.dropout = nn.Dropout(0.3)
#         self.classifier = nn.Linear(self.record_bert.config.hidden_size * 3, num_labels)
    
#     def forward(self, record_input_ids, record_attention_mask, mri_input_ids, mri_attention_mask, ct_input_ids, ct_attention_mask):
#         record_outputs = self.record_bert(input_ids=record_input_ids, attention_mask=record_attention_mask)
#         mri_outputs = self.mri_bert(input_ids=mri_input_ids, attention_mask=mri_attention_mask)
#         ct_outputs = self.ct_bert(input_ids=ct_input_ids, attention_mask=ct_attention_mask)
        
#         # We use the [CLS] token representation
#         record_cls = record_outputs.last_hidden_state[:, 0, :]
#         mri_cls = mri_outputs.last_hidden_state[:, 0, :]
#         ct_cls = ct_outputs.last_hidden_state[:, 0, :]
        
#         combined_cls = torch.cat((record_cls, mri_cls, ct_cls), dim=1)
#         dropout_output = self.dropout(combined_cls)
#         logits = self.classifier(dropout_output)
        
#         return logits


class MultiInputBERTClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(MultiInputBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        
        self.dropout = nn.Dropout(0.3)
        # self.classifier = nn.Linear(self.bert.config.hidden_size * 3, num_labels)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, record_input_ids, record_attention_mask, mri_input_ids, mri_attention_mask, ct_input_ids, ct_attention_mask):
        record_outputs = self.bert(input_ids=record_input_ids, attention_mask=record_attention_mask)
        # mri_outputs = self.bert(input_ids=mri_input_ids, attention_mask=mri_attention_mask)
        # ct_outputs = self.bert(input_ids=ct_input_ids, attention_mask=ct_attention_mask)
        
        # We use the [CLS] token representation
        record_cls = record_outputs.last_hidden_state[:, 0, :]
        # mri_cls = mri_outputs.last_hidden_state[:, 0, :]
        # ct_cls = ct_outputs.last_hidden_state[:, 0, :]
        
        # combined_cls = torch.cat((record_cls, mri_cls, ct_cls), dim=1)
        dropout_output = self.dropout(record_cls)
        logits = self.classifier(record_cls)
        
        return logits


def initialize_model(model_name, device, num_labels=4):
    model = MultiInputBERTClassifier(model_name, num_labels)
    model.to(device)
    return model


