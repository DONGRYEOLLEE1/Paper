from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class BERTDataset:
    def __init__(self, texts, labels, max_len = 128):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.num_examples = len(self.texts)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        tokenized_text = self.tokenizer(text, 
                                        add_special_tokens = True,
                                        padding = 'max_length',
                                        truncation = True,
                                        max_length = self.max_len)
        ids = tokenized_text['input_ids']
        mask = tokenized_text['attention_mask']
        token_type_ids = tokenized_text['token_type_ids']
        
        return {'ids' : torch.tensor(ids, dtype = torch.long), 
                'mask' : torch.tensor(mask, dtype = torch.long),
                'token_type_ids' : torch.tensor(token_type_ids, dtype = torch.long),
                'target' : torch.tensor(label, dtype = torch.long)
                }
        

class ToxicModel(nn.Module):
    """ A simple bert model for training a 2 class classification model"""
    def __init__(self):
        super(ToxicModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', return_dict = False) 
        self.dropout = nn.Dropout(.1)
        self.classifier = nn.Linear(768, 2)
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
def train(model, train_dataset, val_dataset, epochs = 1):
    """train a model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params' : [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay' : 0.01
        },
        {
            'params' : [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay' : 0.0
        }
    ]
    
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr = 5e-5)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_dataset:
            optimizer.zero_grad()
            
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            target = batch['target'].to(device)
                        
            logtis = model(ids, token_type_ids, mask)
            loss = criterion(logtis, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch['ids'].size(0)
            
        train_loss = train_loss / len(train_dataset)
        model.eval()
        valid_loss = 0
        for batch in val_dataset:
            
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            target = batch['target'].to(device)
            
            logtis = model(ids, token_type_ids, mask)
            loss = criterion(logtis, target)
            valid_loss += loss.item() * batch['ids'].size(0)
            
        valid_loss = valid_loss / len(val_dataset)
        print(
            f'Epoch {epoch+1} / {epochs}..'
            f'Train Loss: {train_loss:.3f}..'
            f'Validation Loss: {valid_loss:.3f}..'
        )
        

if __name__ == "__main__":
    df_train = pd.read_csv('D:/lee.dr/Pytorch/TOXIC_CLS/train.csv').head(5000)
    df_valid = pd.read_csv('D:/lee.dr/Pytorch/TOXIC_CLS/test.csv').head(100)
    df_valid_label = pd.read_csv('D:/lee.dr/Pytorch/TOXIC_CLS/test_labels.csv')[['id', 'toxic']]    
    df_valid = pd.merge(df_valid, df_valid_label, on = 'id', how = 'left').reset_index(drop = True)
    
    train_dataset = BERTDataset(df_train.comment_text.values,
                                df_train.toxic.values)
    valid_dataset=  BERTDataset(df_valid.comment_text.values,
                                df_valid.toxic.values)
    
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = 32,
        shuffle = True
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = 32,
        shuffle = True
    )
    model = ToxicModel()
    train(model, train_data_loader, valid_data_loader, epochs = 1)    