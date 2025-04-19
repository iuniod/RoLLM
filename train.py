import torch
import torch.optim as optim
from tqdm import tqdm
from data_utils import prepare_data, split_data
from model import LanguageModel
import wandb
import os
import argparse
from muon import Muon
import torch.distributed as dist

def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
        )
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

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


def lr_finder(model, train_data, optimizer, scaler, param_groups=None, num_iter=100, start_lr=1e-7, end_lr=1e-2):
    """Improved LR finder that:
    - Only operates on specified parameters
    - Has better warmup
    - More stable loss tracking
    """
    model.train()
    original_params = {n: p.detach().clone() for n, p in model.named_parameters()}
    gamma = (end_lr / start_lr) ** (1 / num_iter)

    # Warmup steps (avoid early instability)
    warmup_iters = min(10, num_iter // 10)
    
    lrs = []
    losses = []
    best_lr = start_lr
    min_loss = float('inf')
    
    progress = tqdm(range(num_iter), desc="LR Finder")
    for i in progress:
        # Exponential LR schedule
        lr = start_lr * (gamma ** i)
        
        # Update optimizer LR
        for g in optimizer.param_groups:
            g['lr'] = lr if i >= warmup_iters else start_lr * (i/warmup_iters)
        
        # Get batch and forward/backward
        xb, yb = get_batch(train_data, train_data, 'train')
        optimizer.zero_grad()
        _, loss = model(xb, yb)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Track stats
        current_loss = loss.item()
        lrs.append(lr)
        losses.append(current_loss)
        
        # Update best LR (minimum loss slope)
        if current_loss < min_loss * 0.999:  # Small tolerance
            min_loss = current_loss
            best_lr = lr
        
        progress.set_postfix({
            'lr': f"{lr:.2e}",
            'loss': f"{current_loss:.4f}",
            'best_lr': f"{best_lr:.2e}"
        })
    
    # Restore original model parameters
    for n, p in model.named_parameters():
        p.data.copy_(original_params[n])
    
    return best_lr


def create_optimizer(model, use_muon=False):
    # Create lists for different parameter types
    muon_params = []
    other_params = []
    
    # Categorize parameters more carefully
    print("Categorizing parameters for optimizers...\nTotal parameters: ", sum(p.numel() for p in model.parameters()))
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if use_muon and param.ndim >= 2:
            muon_params.append(param)
        else:
            other_params.append(param)
    
    # Create optimizers
    optimizers = []
    if use_muon and muon_params:
        print(f"Using Muon for {len(muon_params)} parameters")
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        optimizers.append(Muon(muon_params, lr=0.02, momentum=0.95, rank=rank, world_size=world_size))
    if other_params:
        print(f"Using AdamW for {len(other_params)} parameters")
        adam_optimizer = optim.AdamW(other_params, lr=learning_rate)
        optimal_lr = lr_finder(
            model=model,
            train_data=train_data,
            optimizer=adam_optimizer,
            scaler=scaler,
            param_groups=other_params  # Only optimize these params during LR find
        )
        for g in adam_optimizer.param_groups:
            g['lr'] = optimal_lr
        print(f"Found optimal LR: {optimal_lr:.2e}")
        optimizers.append(adam_optimizer)
    
    # Determine if we should use gradient scaling - only if Muon is not used
    use_grad_scaler = not use_muon or not muon_params

    return optimizers, use_grad_scaler


def train(model, optimizers, train_data, val_data, device, scaler, max_iters, save_path="model.pth", use_grad_scaler=False):
    """Trains the model and logs metrics to wandb."""
    model.to(device)
    wandb.init(project="training", name="Model Training")

    for iter in tqdm(range(max_iters), desc='Training'):
        # Periodic evaluation
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, device)
            wandb.log({"train_loss": losses['train'], "val_loss": losses['val']})

        # Get a batch of data
        xb, yb = get_batch(train_data, val_data, 'train')

        # Forward pass
        _, loss = model(xb, yb)
        
        # Backward pass
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        
        if use_grad_scaler:
            scaler.scale(loss).backward()
            for opt in optimizers:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            for opt in optimizers:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

    # Save the model checkpoint
    torch.save(model.state_dict(), save_path)
    wandb.save(save_path)
    wandb.finish()

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train the RoLLM model.")
    parser.add_argument("--use_muon", action="store_true", help="Use Muon optimizer for ≥2D parameters.")
    parser.add_argument("--use_rope", action="store_true", help="Use Rotary Positional Embedding.")
    args = parser.parse_args()

    # Initialize distributed training if using Muon
    if args.use_muon:
        setup_distributed()

    # Initialize Weights & Biases
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    print(f"Using device: {device}")

    scaler = torch.amp.GradScaler('cuda')

    data, vocab_size, encode, decode = prepare_data('./datasets/ro_part_00000_cleaned.parquet')
    train_data, val_data = split_data(data)

    model = LanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout, use_rope=args.use_rope).to(device)
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

    # Training
    optimizers, use_grad_scaler = create_optimizer(model, use_muon=args.use_muon)
    train(model, optimizers, train_data, val_data, device, scaler, max_iters, use_grad_scaler=use_grad_scaler)

    # Generate some text
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
