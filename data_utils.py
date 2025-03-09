import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer

def load_data(file_path):
    cleaned_texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            cleaned_texts.append(line.strip())
    return cleaned_texts

def prepare_data(file_path, model_name="gpt2"):
    cleaned_texts = load_data(file_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    encoded_texts = [torch.tensor(tokenizer.encode(text), dtype=torch.long) for text in tqdm(cleaned_texts, desc='Encoding texts')]
    data = torch.cat(encoded_texts)

    return data, tokenizer.vocab_size, tokenizer.encode, tokenizer.decode

def split_data(data, train_ratio=0.9):
    n = int(train_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data
