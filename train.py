from typing import Tuple, List, Union, Optional, Sequence, Dict
import copy
import os
import argparse
import torch.distributed as dist
import torch
import wandb
import numpy as np
from torch import optim
from tqdm import tqdm
from muon import Muon
from data_utils import prepare_data, split_data
from transformers import GPT2Model, GPT2Config
from model import LanguageModel

def setup_distributed() -> None:
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
        )
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# hyperparameters
BATCH_SIZE = 16
BLOCK_SIZE = 256
MAX_ITERS = 50000
EVAL_INTERVAL = 250
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200
N_EMBD = 1280
N_HEAD = 20
N_LAYER = 36
DROPOUT = 0.0
OPTIMAL_ADAM_LR = None
OPTIMAL_MUON_LR = None

torch.manual_seed(1337)

def get_batch(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    split: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix]).to(DEVICE, non_blocking=True)
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix]).to(DEVICE, non_blocking=True)

    return x, y

@torch.no_grad()
def estimate_loss(
    model: LanguageModel,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    device: Union[str, torch.device]
) -> Dict[str, torch.Tensor]:
    out = {}
    model.eval()
    losses = torch.zeros(EVAL_ITERS, device=device)

    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(train_data, val_data, split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()

    return out

def find_lr(
    model: LanguageModel,
    optimizer: Union[optim.Optimizer, Muon],
    train_data: torch.Tensor,
    start_lr: float,
    end_lr: float,
    num_iter: int = 300,
    smooth_beta: float = 0.9
) -> Tuple[np.ndarray, np.ndarray]:
    # Save initial state
    init_model = copy.deepcopy(model.state_dict())
    init_opt   = copy.deepcopy(optimizer.state_dict())

    mult = (end_lr / start_lr) ** (1 / num_iter)
    warmup_iters = max(1, int(0.1 * num_iter))

    lr = start_lr
    for g in optimizer.param_groups:
        g['lr'] = lr

    model.train()
    avg_loss = 0.0
    lrs, losses = [], []

    for it in tqdm(range(num_iter), desc="LR sweep"):
        x, y = get_batch(train_data, train_data, 'train')
        # Compute this iteration's learning rate
        if it < warmup_iters:
            lr = start_lr * ((it + 1) / warmup_iters)
        else:
            lr = start_lr * (mult ** it)

        for g in optimizer.param_groups:
            g['lr'] = lr

        # Now do the usual forward/backward
        optimizer.zero_grad()

        _, loss = model(x, y)
        loss_v = loss.item()

        # Smooth
        avg_loss = smooth_beta * avg_loss + (1 - smooth_beta) * loss_v
        sm = avg_loss / (1 - smooth_beta ** (it + 1))

        lrs.append(lr)
        losses.append(sm)

        loss.backward()
        optimizer.step()

        lr *= mult
        for g in optimizer.param_groups:
            g['lr'] = lr

    # Restore
    model.load_state_dict(init_model)
    optimizer.load_state_dict(init_opt)

    return np.array(lrs), np.array(losses)


def get_optimal_lr(
    lrs: np.ndarray,
    losses: np.ndarray,
    smoothing_window: int = 5,
    factor: float = 5.0
) -> float:
    # Smooth and compute gradient
    if smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        losses = np.convolve(losses, kernel, mode='same')
    grads = np.gradient(losses, np.log10(lrs))
    idx   = np.nanargmin(grads)
    chosen = lrs[idx] / factor
    print(f"Steepest slope at LR={lrs[idx]:.2e}; selected={chosen:.2e} (factor={factor})")

    return float(chosen)


def find_optimal_lr_adam(
    model: LanguageModel,
    train_data: torch.Tensor,
    params: Sequence[torch.Tensor],
    start_lr: float = 1e-7,
    end_lr: float = 1e-2,
    num_iter: int = 100,
    smoothing_window: int = 1,
    factor: float = 1.0
) -> float:
    opt = optim.AdamW(params, lr=start_lr, weight_decay=0.0)
    lrs, losses = find_lr(model, opt, train_data, start_lr, end_lr, num_iter)

    return get_optimal_lr(lrs, losses, smoothing_window, factor)


def find_optimal_lr_muon(
    model: LanguageModel,
    train_data: torch.Tensor,
    params: Sequence[torch.Tensor],
    start_lr: float = 1e-6,
    end_lr: float = 1e-2,
    num_iter: int = 150,
    smoothing_window: int = 3,
    factor: float = 5.0
) -> float:
    opt = Muon(params, lr=start_lr, momentum=0.0, rank=0, world_size=1)
    lrs, losses = find_lr(model, opt, train_data, start_lr, end_lr, num_iter)

    return get_optimal_lr(lrs, losses, smoothing_window, factor)

def create_optimizer(
    model: LanguageModel,
    train_data: torch.Tensor,
    use_muon: bool = False,
    optimal_adam_lr: Optional[float] = OPTIMAL_ADAM_LR,
    optimal_muon_lr: Optional[float] = OPTIMAL_MUON_LR
) -> Tuple[List[Union[optim.Optimizer, Muon]], bool]:
    # Create lists for different parameter types
    muon_params = []
    other_params = []

    print(
        "Categorizing parameters for optimizers...\nTotal parameters: ",
        sum(p.numel() for p in model.parameters())
    )
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if use_muon and param.ndim >= 2:
            muon_params.append(param)
        else:
            other_params.append(param)

    # Create optimizers
    optimizers = []
    train_data_subset = train_data[:BLOCK_SIZE * 2048]

    if use_muon and muon_params:
        print(f"Using Muon for {len(muon_params)} parameters")

        if optimal_muon_lr is None:
            optimal_muon_lr = find_optimal_lr_muon(model, train_data_subset, muon_params)
        muon_opt = Muon(muon_params, lr=optimal_muon_lr, momentum=0.95, rank=0, world_size=1)
        for g in muon_opt.param_groups:
            g['lr'] = optimal_muon_lr
        print(f"Optimal Muon LR ≈ {optimal_muon_lr:.2e}")
        optimizers.append(muon_opt)

    if other_params:
        print(f"Using AdamW for {len(other_params)} parameters")

        if optimal_adam_lr is None:
            optimal_adam_lr = find_optimal_lr_adam(model, train_data_subset, other_params)
        adam_optimizer = optim.AdamW(other_params, lr=optimal_adam_lr)
        for g in adam_optimizer.param_groups:
            g['lr'] = optimal_adam_lr
        print(f"Optimal Adam LR ≈ {optimal_adam_lr:.2e}")
        optimizers.append(adam_optimizer)

    # Determine if we should use gradient scaling - only if Muon is not used
    use_grad_scaler = not use_muon or not muon_params

    return optimizers, use_grad_scaler


def train(
    model: LanguageModel,
    optimizers: List[Union[optim.Optimizer, Muon]],
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    device: Union[str, torch.device],
    scaler: torch.cuda.amp.GradScaler,
    max_iters: int,
    save_path: str = "model.pth",
    use_grad_scaler: bool = False
) -> None:
    """Trains the model and logs metrics to wandb."""
    wandb.init(project="training_1B", name="Model Training")

    for iter in tqdm(range(max_iters), desc='Training'):
        # Periodic evaluation
        if iter % EVAL_INTERVAL == 0 or iter == max_iters - 1:
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
    parser.add_argument(
        "--use_muon", action="store_true", help="Use Muon optimizer for ≥2D parameters."
    )
    parser.add_argument("--use_rope", action="store_true", help="Use Rotary Positional Embedding.")
    parser.add_argument("--use_unet_skip", action="store_true", help="Use U-Net skip connections.")
    parser.add_argument(
        "--use_tokenmonster", action="store_true", help="Use TokenMonster for tokenization."
    )
    parser.add_argument("--use_gpt2", action="store_true", help="Use GPT-2 model for training. All other options will be ignored.")
    args = parser.parse_args()

    if args.use_gpt2:
        print("Using GPT-2 model. All other options will be ignored.")
        args.use_rope = False
        args.use_unet_skip = False
        args.use_tokenmonster = False
        args.use_muon = False

    # Initialize distributed training if using Muon
    if args.use_muon:
        setup_distributed()

    # Initialize Weights & Biases
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    print(f"Using device: {DEVICE}")

    scaler = torch.amp.GradScaler('cuda')

    # Prepare data for GPT-2 tokenizer or TokenMonster
    if args.use_tokenmonster:
        data, vocab_size, encode, decode = prepare_data(
            './datasets/ro_part_00000_cleaned.parquet', use_tokenmonster=True,
            pretrained_model_name_or_path="./tokenmonster/file.vocab"
        )
    else:
        data, vocab_size, encode, decode = prepare_data('./datasets/ro_part_00000_cleaned.parquet')
    train_data, val_data = split_data(data)

    model = None
    if args.use_gpt2:
        # Use GPT-2 model
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=BLOCK_SIZE,
            n_embd=N_EMBD,
            n_layer=N_LAYER,
            n_head=N_HEAD,
            resid_pdrop=DROPOUT
        )
        model = GPT2Model(config).to(DEVICE)
    else:
        model = LanguageModel(
            vocab_size, N_EMBD, BLOCK_SIZE, N_HEAD, N_LAYER, DROPOUT,
            use_rope=args.use_rope, use_unet_skip=args.use_unet_skip
        ).to(DEVICE)

    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

    # Training
    optimizers, use_grad_scaler = create_optimizer(model, train_data, use_muon=args.use_muon)
    train(
        model, optimizers, train_data, val_data,
        DEVICE, scaler, MAX_ITERS,
        use_grad_scaler=use_grad_scaler
    )
