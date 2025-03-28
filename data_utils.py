import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer
import pandas as pd

ENCODED_TEXTS_PATH = 'encoded_texts.pt'

def prepare_data(file_path, model_name="gpt2"):
    dp = pd.read_parquet(file_path)
    texts = dp['text'].tolist()
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    if ENCODED_TEXTS_PATH:
        try:
            data = torch.load(ENCODED_TEXTS_PATH)
        except FileNotFoundError:
            encoded_texts = [
                torch.tensor(tokenizer.encode(text, truncation=True, max_length=1024), dtype=torch.long)
                for text in tqdm(texts, desc='Encoding texts')
            ]
            data = torch.cat(encoded_texts)

            torch.save(data, ENCODED_TEXTS_PATH)

    return data, tokenizer.vocab_size, tokenizer.encode, tokenizer.decode


def split_data(data, train_ratio=0.9):
    n = int(train_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data
