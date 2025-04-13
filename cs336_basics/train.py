# add to path
import sys
sys.path.append('/users/christineye/cs336/assignment1-basics/cs336_basics')
import torch
import numpy as np
import math
import random

def CELoss(logits, targets):
    """Compute cross-entropy loss for a batch of logits and targets.
    Subtract maximum logit value to prevent overflow."""
    # we will take the softmax over the last dimension (vocab_size)
    # we will return the loss averaged over the batch/seq_len dimensions

    # get maximum along each softmax dimension
    offset = torch.max(logits, dim = -1, keepdim = True)[0]

    # subtract largest element
    logits = logits - offset

    # apply softmax
    xs = torch.gather(logits, dim = -1, index = targets.unsqueeze(-1)).squeeze(-1)
    xs -= torch.log(torch.sum(torch.exp(logits), dim = -1, keepdim = False))

    return -1 * torch.mean(xs)

def cosine_annealing(t, alpha_max, alpha_min, T_w, T_c):
    """Cosine annealing learning rate scheduler.
    t: current iteration
    alpha_max: maximum learning rate
    alpha_min: minimum learning rate
    T_w: warmup period
    T_c: cosine annealing period
    """
    if t < T_w:
        return t * alpha_max / T_w
    if t <= T_c:
        return alpha_min + 0.5 * (1 + math.cos((t - T_w) * math.pi / (T_c - T_w))) * (alpha_max - alpha_min)

    return alpha_min

def gradient_clipping(params, max_l2, eps = 1e-6):
    """Run gradient clipping, modifying in-place.
    params: Iterable of torch.nn.Parameter objects
    max_l2: maximum L2 norm of the gradient
    eps: small constant to prevent division by zero
    """
    grad = [p.grad for p in params if p.grad is not None]
    grads_flat = torch.stack([g.detach().flatten() for g in grad])
    
    l2_norm = torch.norm(grads_flat, 2)
    if l2_norm <= max_l2:
        return
    else:
        coef = max_l2 / (l2_norm + eps)
        for g in grad:
            g *= coef

def load_data(x: np.array, batch_size: int, seq_len: int, device: str) -> torch.Tensor:
    """Load a batch from the input array x, sampling sequences independently.
    x: input array of shape length x 1
    batch_size: number of sequences to sample
    seq_len: length of each sequence
    device: device to load the data onto
    """
    # assume x is of shape length x 1
    length = x.shape[0]
    start_indices = random.sample(range(0, length - seq_len), batch_size)
    batch = np.zeros((batch_size, seq_len))
    targets = np.zeros((batch_size, seq_len))
    
    for i, p in enumerate(start_indices):
        batch[i, :] = x[p: p + seq_len]
        targets[i, :] = x[p + 1: p + 1 + seq_len]
    
    return torch.Tensor(batch).to(device), torch.Tensor(targets).to(device)

def save_checkpoint(model, optimizer, iteration, out):
    to_save = {"model": model.state_dict(),
               "optimizer": optimizer.state_dict(),
               "iteration": iteration}
    torch.save(to_save, out)

def load_checkpoint(src, model, optimizer):
    loaded = torch.load(src)
    model.load_state_dict(loaded["model"])
    optimizer.load_state_dict(loaded["optimizer"])
    return loaded["iteration"]