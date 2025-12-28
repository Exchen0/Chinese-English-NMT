import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import os
import json
import math

from models.utils import load_data, tokenize_chinese, tokenize_english, build_vocab, clean_text
from models.transformer_model import TransformerSeq2Seq
from models.rnn_model import RNNEncoder, RNNDecoder, RNNSeq2Seq

def collate_fn(batch, src_pad_idx, trg_pad_idx):
    src_batch = [item[0] for item in batch]
    trg_batch = [item[1] for item in batch]
    
    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=src_pad_idx)
    trg_padded = nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=trg_pad_idx)
    
    return src_padded, trg_padded

class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, trg_vocab, max_len=128):
        self.data = data
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text = clean_text(self.data[idx]['zh'])
        trg_text = clean_text(self.data[idx]['en'])
        
        src_tokens = tokenize_chinese(src_text)[:self.max_len]
        trg_tokens = tokenize_english(trg_text)[:self.max_len]

        src_indices = [self.src_vocab['<SOS>']] + \
                      [self.src_vocab.get(t, self.src_vocab['<UNK>']) for t in src_tokens] + \
                      [self.src_vocab['<EOS>']]
        
        trg_indices = [self.trg_vocab['<SOS>']] + \
                      [self.trg_vocab.get(t, self.trg_vocab['<UNK>']) for t in trg_tokens] + \
                      [self.trg_vocab['<EOS>']]

        return torch.tensor(src_indices), torch.tensor(trg_indices)

def get_lr_scheduler(optimizer, warmup_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(warmup_steps) / float(max(1, current_step))) ** 0.5
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(model, train_loader, valid_loader, config, src_vocab, trg_vocab):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    is_rnn = config['model']['type'] == 'rnn'
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], betas=(0.9, 0.98), eps=1e-9)
    
    scheduler = get_lr_scheduler(optimizer, config['training']['warmup_steps'])
    
    criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab['<PAD>'], label_smoothing=0.1)

    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    best_valid_loss = float('inf')

    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config['training']['epochs']}]")

        for src, trg in train_bar:
            src, trg = src.to(device), trg.to(device)
 
            trg_input = trg[:, :-1]
            trg_real = trg[:, 1:].contiguous().view(-1)
            
            optimizer.zero_grad()
            if is_rnn:
                tf_ratio = config['training'].get('teacher_forcing_ratio')
                output = model(src, trg, teacher_forcing_ratio=tf_ratio)
            else:
                output = model(src, trg_input) # [batch, seq_len-1, vocab_size]
            
            loss = criterion(output.view(-1, output.shape[-1]), trg_real)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            
            optimizer.step()
            scheduler.step() 

            total_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.7f}")

        avg_train_loss = total_loss / len(train_loader)
        avg_valid_loss = validate(model, valid_loader, criterion, device, trg_vocab)
        
        print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_valid_loss:.4f}")

        lr = config['training']['learning_rate']
        bs = config['data']['batch_size']
        norm = config['model']['transformer']['normalization']
        pe = config['model']['transformer']['position_embedding']
        attn = config['model']['rnn']['attention']
        teacher = config['training']['teacher_forcing_ratio']

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_model_name = f"best_model_rnn_lr{lr}_bs{bs}_{attn}_{teacher}.pt"
            best_path = os.path.join(save_dir, best_model_name)
            torch.save(model.state_dict(), best_path)
            print(f"🌟 Best model saved as: {best_model_name}")
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(save_dir, 'latest_model.pt'))

def validate(model, valid_loader, criterion, device, trg_vocab):
    model.eval()
    total_loss = 0.0
    is_rnn = isinstance(model, RNNSeq2Seq)
    
    with torch.no_grad():
        for src, trg in valid_loader:
            src, trg = src.to(device), trg.to(device)
            
            trg_real = trg[:, 1:].contiguous().view(-1)
            
            if is_rnn:
                output = model(src, trg, teacher_forcing_ratio=0.0)
            else:
                trg_input = trg[:, :-1]
                output = model(src, trg_input)

            output_dim = output.shape[-1]
            loss = criterion(output.view(-1, output_dim), trg_real)
            total_loss += loss.item()
            
    return total_loss / len(valid_loader)

if __name__ == "__main__":
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_raw = load_data(config['data']['train_file'])
    valid_raw = load_data(config['data']['valid_file'])

    src_vocab = build_vocab([x['zh'] for x in train_raw], tokenize_chinese, min_freq=2)
    trg_vocab = build_vocab([x['en'] for x in train_raw], tokenize_english, min_freq=2)

    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/src_vocab.json", "w") as f: json.dump(src_vocab, f)
    with open("checkpoints/trg_vocab.json", "w") as f: json.dump(trg_vocab, f)

    train_ds = TranslationDataset(train_raw, src_vocab, trg_vocab, config['data']['max_len'])
    valid_ds = TranslationDataset(valid_raw, src_vocab, trg_vocab, config['data']['max_len'])
    
    train_loader = DataLoader(train_ds, batch_size=config['data']['batch_size'], shuffle=True, 
                              collate_fn=lambda b: collate_fn(b, src_vocab['<PAD>'], trg_vocab['<PAD>']))
    valid_loader = DataLoader(valid_ds, batch_size=config['data']['batch_size'], 
                              collate_fn=lambda b: collate_fn(b, src_vocab['<PAD>'], trg_vocab['<PAD>']))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = config['model']['type'] 
    if model_type == 'transformer':
        print("Initializing Transformer Model...")
        model = TransformerSeq2Seq(
            input_dim=len(src_vocab), 
            output_dim=len(trg_vocab), 
            emb_dim=config['model']['transformer']['hidden_size'], 
            n_layers=config['model']['transformer']['encoder_layers'], 
            n_heads=config['model']['transformer']['num_heads'], 
            ff_dim=config['model']['transformer']['ff_size'], 
            dropout=config['model']['transformer']['dropout'],
            position_embedding=config['model']['transformer']['position_embedding'],
            normalization=config['model']['transformer']['normalization']
        )
    
    elif model_type == 'rnn':
        print(f"Initializing RNN Model ({config['model']['rnn']['rnn_type']})...")
        encoder = RNNEncoder(
            input_dim=len(src_vocab),
            emb_dim=config['model']['transformer']['hidden_size'], 
            hidden_size=config['model']['rnn']['hidden_size'],
            n_layers=config['model']['rnn']['n_layers'],
            rnn_type=config['model']['rnn']['rnn_type'],
            dropout=config['model']['transformer']['dropout']
        )
        decoder = RNNDecoder(
            output_dim=len(trg_vocab),
            emb_dim=config['model']['transformer']['hidden_size'],
            hidden_size=config['model']['rnn']['hidden_size'],
            n_layers=config['model']['rnn']['n_layers'],
            rnn_type=config['model']['rnn']['rnn_type'],
            attn_method=config['model']['rnn']['attention'],
            dropout=config['model']['transformer']['dropout']
        )
        model = RNNSeq2Seq(encoder, decoder, device)

    train(model, train_loader, valid_loader, config, src_vocab, trg_vocab)