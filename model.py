import torch
import torch.nn as nn
from torch.nn import functional as F
from rotary_embedding_torch import RotaryEmbedding

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout, use_rope=False):
        super().__init__()
        self.use_rope = use_rope
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

        # rotary positional encoder for this head
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(dim = head_size, use_xpos = True)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        if self.use_rope:
            # apply rotary position embeddings to both q and k
            q, k = self.rotary_emb.rotate_queries_and_keys(q, k)
            # now q, k each still have shape (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)

        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, dropout, use_rope=False):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout, use_rope) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout, use_rope=False):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout, use_rope)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout, use_rope=False, use_unet_skip=False):
        super().__init__()
        self.use_rope = use_rope
        self.use_unet_skip = use_unet_skip
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        if not self.use_rope:
            self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Transformer blocks: either sequential or ModuleList with U-Net skips
        if self.use_unet_skip:
            assert n_layer % 2 == 0, "Number of layers must be even for U-Net skipping"
            # create blocks as ModuleList for indexed forward
            self.blocks = nn.ModuleList([
                Block(n_embd, n_head, block_size, dropout, use_rope)
                for _ in range(n_layer)
            ])
            # one learnable weight per skip pair
            self.skip_weights = nn.Parameter(torch.zeros(n_layer // 2))
        else:
            # standard sequential stacking
            self.blocks = nn.Sequential(*[
                Block(n_embd, n_head, block_size, dropout, use_rope)
                for _ in range(n_layer)
            ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)

        if self.use_rope:
            x = tok_emb
        else:
            pos = torch.arange(T, device=idx.device)
            x = tok_emb + self.position_embedding_table(pos)

        # apply transformer blocks
        if self.use_unet_skip:
            skip_stack = []
            half = len(self.blocks) // 2
            for i, block in enumerate(self.blocks):
                if i >= half:
                    # add skip from corresponding early layer
                    x = x + self.skip_weights[i - half] * skip_stack.pop()
                x = block(x)
                if i < half:
                    # store for later skip
                    skip_stack.append(x)
        else:
            # simple sequential application
            x = self.blocks(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
