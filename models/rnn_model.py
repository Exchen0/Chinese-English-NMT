import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class GlobalAttention(nn.Module):
    def __init__(self, hidden_size, method='additive'):
        super(GlobalAttention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'multiplicative':
            self.Wa = nn.Linear(hidden_size, hidden_size)
        elif self.method == 'additive':
            self.Wa = nn.Linear(hidden_size, hidden_size)
            self.Ua = nn.Linear(hidden_size, hidden_size)
            self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # query: [batch, 1, hidden_size] (decoder current hidden)
        # keys: [batch, seq_len, hidden_size] (encoder outputs)
        
        if self.method == 'dot':
            scores = torch.bmm(query, keys.transpose(1, 2)) # [batch, 1, seq_len]
        elif self.method == 'multiplicative':
            scores = torch.bmm(query, self.Wa(keys).transpose(1, 2))
        elif self.method == 'additive':
            # Bahdanau style
            q = self.Wa(query) # [batch, 1, hidden_size]
            k = self.Ua(keys)  # [batch, seq_len, hidden_size]
            scores = self.Va(torch.tanh(q + k)).transpose(1, 2) # [batch, 1, seq_len]

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys) # [batch, 1, hidden_size]
        return context, weights

class RNNEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_size, n_layers=2, rnn_type='GRU', dropout=0.1):
        super(RNNEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn_type = rnn_type
        
        rnn_cell = nn.GRU if rnn_type == 'GRU' else nn.LSTM
        self.rnn = rnn_cell(emb_dim, hidden_size, num_layers=n_layers, 
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class RNNDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_size, n_layers=2, 
                 rnn_type='GRU', attn_method='additive', dropout=0.1):
        super(RNNDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = GlobalAttention(hidden_size, method=attn_method)
        
        rnn_cell = nn.GRU if rnn_type == 'GRU' else nn.LSTM
        self.rnn = rnn_cell(emb_dim + hidden_size, hidden_size, num_layers=n_layers, 
                            batch_first=True, dropout=dropout)
        
        self.fc_out = nn.Linear(hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_token, hidden, encoder_outputs):
        # input_token: [batch, 1]
        embedded = self.dropout(self.embedding(input_token)) # [batch, 1, emb_dim]

        query = hidden[-1].unsqueeze(1) if self.rnn_type == 'GRU' else hidden[0][-1].unsqueeze(1)
        
        context, attn_weights = self.attention(query, encoder_outputs)
        
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        prediction = self.fc_out(torch.cat((output, context), dim=2))
        return prediction, hidden, attn_weights

# --- Seq2Seq Wrapper ---
class RNNSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(RNNSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # trg: [batch, trg_len]
        batch_size = src.size(0)
        trg_len = trg.size(1)
        
        encoder_outputs, hidden = self.encoder(src)
        
        decoder_input = trg[:, 0].unsqueeze(1) 
        outputs = []

        for t in range(1, trg_len):
            output, hidden, _ = self.decoder.forward_step(decoder_input, hidden, encoder_outputs)
            outputs.append(output)
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(2)
            decoder_input = trg[:, t].unsqueeze(1) if teacher_force else top1
            
        return torch.cat(outputs, dim=1) # [batch, trg_len-1, output_dim]