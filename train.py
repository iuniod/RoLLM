import torch
import torch.optim as optim
from tqdm import tqdm
from data_utils import prepare_data, split_data
from model import LanguageModel
import wandb
import os

# hyperparameters
batch_size = 16
block_size = 32
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.1

torch.manual_seed(1337)

def get_batch(train_data, val_data, split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == "__main__":
    # Initialize Weights & Biases
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(project="language_model_training", config={
        "batch_size": batch_size,
        "block_size": block_size,
        "max_iters": max_iters,
        "eval_interval": eval_interval,
        "learning_rate": learning_rate,
        "device": device,
        "eval_iters": eval_iters,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "dropout": dropout
    })

    data, vocab_size, encode, decode = prepare_data('./datasets/ro_part_00000_cleaned.parquet')
    train_data, val_data = split_data(data)

    model = LanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout).to(device)
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in tqdm(range(max_iters), desc='Training'):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            # Log losses to wandb
            losses = estimate_loss(model, train_data, val_data)
            wandb.log({"train_loss": losses['train'], "val_loss": losses['val'], "step": iter})

        xb, yb = get_batch(train_data, val_data, 'train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Log batch loss to wandb
        wandb.log({"batch_loss": loss.item(), "step": iter})

    # Save the final model
    torch.save(model.state_dict(), "final_model.pth")
    wandb.save("final_model.pth")

    # Generate some text
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))

    # Finish wandb run
    wandb.finish()
