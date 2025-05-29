from typing import Tuple, Callable, List
from tqdm import tqdm
from transformers import GPT2Tokenizer
import torch
import pandas as pd

ENCODED_TEXTS_PATH = 'encoded_texts.pt'
CHUNK_SIZE = 1024

def prepare_data(
    file_path: str,
    model_name: str ="gpt2"
) -> Tuple[
    torch.Tensor,
    int,
    Callable[[str], List[int]],
    Callable[[List[int]], str]
]:
    dp = pd.read_parquet(file_path)
    texts = dp['text'].tolist()
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    if ENCODED_TEXTS_PATH:
        try:
            data = torch.load(ENCODED_TEXTS_PATH)
        except FileNotFoundError:
            encoded_texts = []

            for text in tqdm(texts, desc="Encoding texts"):
                encoded = tokenizer.encode(text, truncation=False)
                chunks = [
                    torch.tensor(encoded[i : i + CHUNK_SIZE], dtype=torch.long)
                    for i in range(0, len(encoded), CHUNK_SIZE)
                ]
                encoded_texts.extend(chunks)

            data = torch.cat(encoded_texts)
            torch.save(data, ENCODED_TEXTS_PATH)

    return data, tokenizer.vocab_size, tokenizer.encode, tokenizer.decode


def split_data(
    data: torch.Tensor,
    train_ratio: float = 0.9
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(train_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data
