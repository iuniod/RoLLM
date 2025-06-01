from typing import Tuple, Callable, List
from tqdm import tqdm
from transformers import GPT2Tokenizer
import torch
import tokenmonster
import pandas as pd

GPT2_CACHE_PATH = 'encoded_texts_gpt2.pt'
TM_CACHE_PATH   = 'encoded_texts_tokenmonster.pt'
CHUNK_SIZE      = 1024

def prepare_data(
    file_path: str,
    use_tokenmonster: bool = False,
    pretrained_model_name_or_path: str = "gpt2"
) -> Tuple[
    torch.Tensor,
    int,
    Callable[[str], List[int]],
    Callable[[List[int]], str]
]:
    # Load the Parquet and grab the "text" column
    df = pd.read_parquet(file_path)
    texts: List[str] = df['text'].tolist()

    # Instantiate either TokenMonster or GPT2
    if use_tokenmonster:
        vocab = tokenmonster.load(pretrained_model_name_or_path)

        def encode_fn(text: str) -> List[int]:
            arr = vocab.tokenize(text)
            return arr.tolist()

        def decode_fn(token_ids: List[int]) -> str:
            return vocab.decode(token_ids)

        vocab_size = vocab.vocab_size
        cache_path = TM_CACHE_PATH

    else:
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path)

        encode_fn = tokenizer.encode
        decode_fn = tokenizer.decode
        vocab_size = tokenizer.vocab_size
        cache_path = GPT2_CACHE_PATH

    # Try loading the cached file. If not found, do the full encode+chunk+save.
    print(f"Loading encoded texts from {cache_path}...")
    try:
        data = torch.load(cache_path)
    except FileNotFoundError:
        encoded_chunks: List[torch.Tensor] = []

        # Encode each text (this can be slow; we show a tqdm bar)
        for text in tqdm(texts, desc="Encoding texts"):
            if text is None or text.strip() == "":
                continue
            token_ids = encode_fn(text)  # str -> List[int]
            # Break into CHUNK_SIZE‐length pieces:
            for i in range(0, len(token_ids), CHUNK_SIZE):
                chunk = token_ids[i : i + CHUNK_SIZE]
                encoded_chunks.append(torch.tensor(chunk, dtype=torch.long))

        # Concatenate all chunks (a list of 1D tensors) into one big 1D tensor
        data = torch.cat(encoded_chunks, dim=0)
        torch.save(data, cache_path)

    return data, vocab_size, encode_fn, decode_fn


def split_data(
    data: torch.Tensor,
    train_ratio: float = 0.9
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(train_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data
