import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import math
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from models.utils import load_data
from tqdm import tqdm
from sacrebleu.metrics import BLEU

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

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

def evaluate_bleu(model_path, test_file, device):
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()

    test_data = load_data(test_file)
    prefix = "translate Chinese to English: "
    
    hypotheses = []
    references = []

    print(f"正在对 {len(test_data)} 条测试数据进行翻译...")
    
    for item in tqdm(test_data):
        src_text = prefix + item['zh']
        ref_text = item['en']

        input_ids = tokenizer(src_text, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_length=128, 
                num_beams=4, 
                early_stopping=True
            )

        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        hypotheses.append(pred_text)
        references.append([ref_text])

    bleu = BLEU()
    result = bleu.corpus_score(hypotheses, references)
    
    print("\n" + "="*30)
    print(f"Final BLEU Score: {result.score:.2f}")
    print(f"Details: {result}")
    print("="*30)

    fluency_eval = FluencyEvaluator(device)
    ppl = fluency_eval.corpus_ppl(hypotheses)
    print(f" Fluency (Perplexity): {ppl:.2f}")

    for i in range(3):
        print(f"Source: {test_data[i]['zh']}")
        print(f"Ref: {test_data[i]['en']}")
        print(f"Hyp: {hypotheses[i]}\n")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = "./t5_best_model"
    TEST_FILE = "/data/250010022/sunxc/workspace/LLM/data/test.jsonl"
    
    evaluate_bleu(MODEL_PATH, TEST_FILE, device)