import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.scale * x / norm

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=512, mode='absolute'):
        super(PositionalEncoding, self).__init__()
        self.emb_dim = emb_dim
        self.mode = mode

        if self.mode == 'absolute':
            pe = torch.zeros(max_len, emb_dim)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * -(math.log(10000.0) / emb_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0)) # [1, max_len, emb_dim]
            
        elif self.mode == 'relative':
            inv_freq = 1.0 / (10000 ** (torch.arange(0, emb_dim, 2).float() / emb_dim))
            self.register_buffer("inv_freq", inv_freq)

    def _apply_rope(self, x):
        seq_len = x.size(1)
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) 
        emb = torch.cat((freqs, freqs), dim=-1) 
        
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        
        x1 = x[..., :self.emb_dim // 2]
        x2 = x[..., self.emb_dim // 2:]
        rotated_x = torch.cat((-x2, x1), dim=-1)
        
        return (x * cos_emb) + (rotated_x * sin_emb)

    def forward(self, x):
        if self.mode == 'absolute':
            return x + self.pe[:, :x.size(1), :] 
        elif self.mode == 'relative':
            return self._apply_rope(x)
        return x

class TransformerSeq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, n_layers, n_heads, ff_dim, dropout, position_embedding, normalization):
        super(TransformerSeq2Seq, self).__init__()
        
        self.emb_dim = emb_dim
        self.src_embedding = nn.Embedding(input_dim, emb_dim)
        self.trg_embedding = nn.Embedding(output_dim, emb_dim)
        self.scaling = math.sqrt(emb_dim) 
        
        self.position_embedding = PositionalEncoding(emb_dim, mode=position_embedding)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim, 
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim, 
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        self.fc_out = nn.Linear(emb_dim, output_dim)

        if normalization == "LayerNorm":
            self.norm = nn.LayerNorm(emb_dim)
        elif normalization == "RMSNorm":
            self.norm = RMSNorm(emb_dim)
        else:
            raise ValueError("Unknown normalization method.")
        
    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, trg):
        src_padding_mask = (src == 0) 
        tgt_padding_mask = (trg == 0)

        src_emb = self.position_embedding(self.src_embedding(src) * self.scaling)
        trg_emb = self.position_embedding(self.trg_embedding(trg) * self.scaling)

        tgt_mask = self.generate_square_subsequent_mask(trg.size(1), trg.device)

        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)

        output = self.decoder(
            trg_emb, 
            memory, 
            tgt_mask=tgt_mask, 
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        return self.fc_out(self.norm(output))