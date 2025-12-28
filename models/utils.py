import json
import jieba
import nltk

from collections import Counter
from sklearn.model_selection import train_test_split

import re

import torch
import torch.nn as nn
import numpy as np

def load_pretrained_embeddings(vocab, embedding_path, embed_dim=300):
    vocab_size = len(vocab)
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embed_dim))

    weight = torch.FloatTensor(embedding_matrix)
    return nn.Embedding.from_pretrained(weight, freeze=False)
    
def clean_text(text):
    text = re.sub(r"[^\u4e00-\u9fa5^a-z^A-Z^0-9，。！？,.!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def filter_sentences(tokenized_sentences, max_len=100):
    return [s[:max_len] for s in tokenized_sentences if len(s) > 0]
def tokenize_chinese(text, tool='jieba'):
    if tool == 'jieba':
        return list(jieba.cut(text))
    else:
        raise ValueError("Unknown Chinese tokenization tool")

def tokenize_english(text, tool='bpe'):
    tokens = nltk.word_tokenize(text)
    return tokens

def build_vocab(texts, tokenizer_func, min_freq=2):
    all_tokens = []
    for text in texts:
        cleaned = clean_text(text)
        tokens = tokenizer_func(cleaned)
        all_tokens.extend(tokens)
    
    counter = Counter(all_tokens)
    tokens = [word for word, freq in counter.items() if freq >= min_freq]
    
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<SOS>": 2,
        "<EOS>": 3
    }
    for i, token in enumerate(tokens):
        vocab[token] = i + 4
        
    return vocab


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    return data
