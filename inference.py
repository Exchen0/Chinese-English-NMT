import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn.functional as F
import json
import yaml
import os
from tqdm import tqdm
from sacrebleu.metrics import BLEU

from models.transformer_model import TransformerSeq2Seq
from models.utils import clean_text, tokenize_english, tokenize_chinese, load_data
from models.rnn_model import RNNEncoder, RNNDecoder, RNNSeq2Seq

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import math

class FluencyEvaluator:
    def __init__(self, device):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def sentence_ppl(self, sentence: str):
        encodings = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )
        input_ids = encodings.input_ids.to(self.device)
        outputs = self.model(input_ids, labels=input_ids)
        loss = outputs.loss
        return math.exp(loss.item())

    @torch.no_grad()
    def corpus_ppl(self, sentences):
        total_nll = 0.0
        total_tokens = 0

        for s in sentences:
            enc = self.tokenizer(
                s,
                return_tensors="pt",
                truncation=True,
                max_length=128
            )
            input_ids = enc.input_ids.to(self.device)

            if input_ids.size(1) < 2:
                continue

            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            n_tokens = input_ids.size(1) - 1
            total_nll += loss.item() * n_tokens
            total_tokens += n_tokens

        return math.exp(total_nll / total_tokens)

def translate_sentence(model, sentence_text, src_vocab, trg_vocab, device, max_len=128):
    model.eval()
    rev_trg_vocab = {int(idx): word for word, idx in trg_vocab.items()}
    

    tokens = tokenize_chinese(clean_text(sentence_text)) 
    
    src_indices = [src_vocab['<SOS>']] + \
                  [src_vocab.get(token, src_vocab['<UNK>']) for token in tokens] + \
                  [src_vocab['<EOS>']]
    
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)

    with torch.no_grad():
        src_padding_mask = (src_tensor == src_vocab['<PAD>'])
        src_emb = model.position_embedding(model.src_embedding(src_tensor) * model.scaling)
        memory = model.encoder(src_emb, src_key_padding_mask=src_padding_mask)

    trg_indices = [trg_vocab['<SOS>']]
    
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)
        with torch.no_grad():
            trg_emb = model.position_embedding(model.trg_embedding(trg_tensor) * model.scaling)
            output = model.decoder(trg_emb, memory, memory_key_padding_mask=src_padding_mask)
            prediction = model.fc_out(output)
        
        next_token = prediction.argmax(2)[:, -1].item()
        if next_token == trg_vocab['<EOS>']:
            break
        trg_indices.append(next_token)

    result_tokens = [rev_trg_vocab[idx] for idx in trg_indices 
                     if idx not in [trg_vocab['<SOS>'], trg_vocab['<EOS>'], trg_vocab['<PAD>']]]
    
    return " ".join(result_tokens) 


def translate_sentence_rnn_beam(model, sentence_text, src_vocab, trg_vocab, device, beam_size=5, max_len=128):
    model.eval()
    rev_trg_vocab = {int(idx): word for word, idx in trg_vocab.items()}
    
    tokens = tokenize_chinese(clean_text(sentence_text)) 
    src_indices = [src_vocab['<SOS>']] + \
                  [src_vocab.get(token, src_vocab['<UNK>']) for token in tokens] + \
                  [src_vocab['<EOS>']]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

    beams = [([trg_vocab['<SOS>']], 0.0, hidden)]
    completed_sequences = []

    for _ in range(max_len):
        new_candidates = []
        
        for seq, score, current_hidden in beams:
            if seq[-1] == trg_vocab['<EOS>']:
                completed_sequences.append((seq, score))
                continue

            decoder_input = torch.LongTensor([seq[-1]]).unsqueeze(0).to(device)
            
            with torch.no_grad():
                prediction, next_hidden, _ = model.decoder.forward_step(
                    decoder_input, current_hidden, encoder_outputs
                )
            
            log_probs = F.log_softmax(prediction.squeeze(1), dim=-1) # [1, vocab_size]

            topk_log_probs, topk_indices = log_probs.topk(beam_size)

            for i in range(beam_size):
                next_token = topk_indices[0][i].item()
                next_score = score + topk_log_probs[0][i].item()
                count = seq.count(next_token)
                if count > 0:
                    next_score -= count * 0.7   

                new_candidates.append(
                    (seq + [next_token], next_score, next_hidden)
                )

        new_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = new_candidates[:beam_size]

        if all(s[-1] == trg_vocab['<EOS>'] for s, _, _ in beams):
            break

    for seq, score, _ in beams:
        completed_sequences.append((seq, score))

    alpha = 0.6
    def normalize_score(item):
        seq, score = item
        return score / (len(seq) ** alpha)

    completed_sequences.sort(key=normalize_score, reverse=True)
    best_seq, _ = completed_sequences[0]

    result_tokens = [rev_trg_vocab[idx] for idx in best_seq 
                     if idx not in [trg_vocab['<SOS>'], trg_vocab['<EOS>'], trg_vocab['<PAD>']]]
    
    return " ".join(result_tokens)

def run_evaluation(model, test_raw, src_vocab, trg_vocab, device):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n Model Parameters: {total_params:,}")
    hypotheses = []
    references = []

    print("Starting Chinese to English Translation...")
    for example in tqdm(test_raw):
        src_sentence = example['zh']
        en_reference = example['en']

        prediction = translate_sentence(
            model, src_sentence, src_vocab, trg_vocab, device
        )

        hypotheses.append(prediction)
        references.append([en_reference])

    # BLEU
    bleu = BLEU()
    bleu_score = bleu.corpus_score(hypotheses, references)
    print(f"\n BLEU: {bleu_score.score:.2f}")

    # ===== Fluency =====
    fluency_eval = FluencyEvaluator(device)
    ppl = fluency_eval.corpus_ppl(hypotheses)
    print(f" Fluency (Perplexity): {ppl:.2f}")

    # Examples
    print("\n Examples:")
    for i in range(5):
        print(f"Source: {test_raw[i]['zh']}")
        print(f"Target: {test_raw[i]['en']}")
        print(f"Predict: {hypotheses[i]}")
        print("-" * 30)

if __name__ == "__main__":
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading vocabularies from JSON...")
    try:
        with open("checkpoints/src_vocab.json", "r") as f:
            src_vocab = json.load(f)
        with open("checkpoints/trg_vocab.json", "r") as f:
            trg_vocab = json.load(f)
    except FileNotFoundError:
        print("Error: Vocab files not found. Please run train.py first.")
        exit()

    model_type = config['model']['type'] # 
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

    model_path = "/data/250010022/sunxc/workspace/LLM/checkpoints/best_model_lr0.0001_bs256_LayerNorm_absolute.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded weights from {model_path}")
    else:
        print("Error: best_model.pt not found.")
        exit()

    model.to(device)

    test_raw = load_data(config['data']['test_file'])

    run_evaluation(model, test_raw, src_vocab, trg_vocab, device)