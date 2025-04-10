import torch
import torch.optim as optim
from tqdm import tqdm
from data_utils import prepare_data, split_data
from model import LanguageModel
from torch.optim.lr_scheduler import ExponentialLR
import wandb
import os

# hyperparameters
batch_size = 16
block_size = 512
max_iters = 50000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 512
n_head = 8
n_layer = 8
dropout = 0.0

torch.manual_seed(1337)

def get_batch(train_data, val_data, split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix]).to(device, non_blocking=True)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device, non_blocking=True)

    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, device):
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters, device=device)

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()

    return out


def lr_finder(model, train_data, optimizer, device, scaler, start_lr=1e-7, end_lr=10, num_iter=1000):
    """Finds the optimal learning rate by gradually increasing it and logging loss."""
    model.train()
    wandb.init(project="learning-rate-finder", name="LR Finder")

    model.to(device)
    optimizer.param_groups[0]['lr'] = start_lr
    scheduler = ExponentialLR(optimizer, gamma=(end_lr/start_lr) ** (1 / num_iter))
    lrs, losses = [], []

    for _ in tqdm(range(num_iter), desc="LR Finder"):
        xb, yb = get_batch(train_data, train_data, 'train')
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())
        wandb.log({"learning_rate": lrs[-1], "loss": losses[-1]})

    wandb.finish()
    return lrs[losses.index(min(losses))] / 10


def train(model, optimizer, train_data, val_data, device, scaler, max_iters=1000, save_path="model.pth"):
    """Trains the model and logs metrics to wandb."""
    model.to(device)
    wandb.init(project="training", name="Model Training")

    for iter in tqdm(range(max_iters), desc='Training'):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, device)
            wandb.log({"train_loss": losses['train'], "val_loss": losses['val'], "step": iter})

        xb, yb = get_batch(train_data, val_data, 'train')
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.float16):
            _, loss = model(xb, yb)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


    # Save the model checkpoint
    torch.save(model.state_dict(), save_path)
    wandb.save(save_path)

    wandb.finish()


if __name__ == "__main__":
    # Initialize Weights & Biases
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    print(f"Using device: {device}")

    scaler = torch.amp.GradScaler('cuda')

    data, vocab_size, encode, decode = prepare_data('./datasets/ro_part_00000_cleaned.parquet')
    train_data, val_data = split_data(data)

    model = LanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout).to(device)
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

    # Find the optimal learning rate
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    optimal_lr = lr_finder(model, train_data, optimizer, device, scaler)
    print(f"Optimal learning rate: {optimal_lr}")

    # Training
    model = LanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=optimal_lr)
    train(model, optimizer, train_data, val_data, device, scaler, max_iters)

    # Generate some text
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
