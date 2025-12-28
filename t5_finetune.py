import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from tqdm import tqdm
from models.utils import load_data

class T5TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.prefix = "translate Chinese to English: "

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text = self.prefix + self.data[idx]['zh']
        trg_text = self.data[idx]['en']

        source = self.tokenizer(src_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        target = self.tokenizer(trg_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        labels = target["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": labels
        }

def validate_t5(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
    return total_loss / len(val_loader)

def train_t5(model, train_loader, val_loader, config, device, save_path):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], eps=1e-8)
    
    num_train_steps = len(train_loader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=num_train_steps)

    best_val_loss = float('inf')

    for epoch in range(config['epochs']):
        model.train()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        total_train_loss = 0
        
        for batch in train_bar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        val_loss = validate_t5(model, val_loader, device)
        print(f"Epoch {epoch+1} - Train Loss: {total_train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"发现更好的模型，正在保存至 {save_path}...")
            for param in model.parameters():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()

            model.save_pretrained(save_path, safe_serialization=True)
            
            tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "google/mt5-small"
    save_model_path = "./t5_best_model"
    
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    train_raw = load_data('/data/250010022/sunxc/workspace/LLM/data/train_100k.jsonl')
    valid_raw = load_data('/data/250010022/sunxc/workspace/LLM/data/valid.jsonl')
    
    train_loader = DataLoader(T5TranslationDataset(train_raw, tokenizer), batch_size=64, shuffle=True)
    val_loader = DataLoader(T5TranslationDataset(valid_raw, tokenizer), batch_size=64)

    config = {'learning_rate': 2e-4, 'epochs': 30, 'warmup_steps': 1000}
    
    train_t5(model, train_loader, val_loader, config, device, save_model_path)